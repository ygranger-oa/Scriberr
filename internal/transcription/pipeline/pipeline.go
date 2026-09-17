package pipeline

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"scriberr/internal/transcription/interfaces"
	"scriberr/pkg/logger"
)

// ProcessingPipeline handles the full processing workflow with preprocessing
type ProcessingPipeline struct {
	preprocessors  []interfaces.Preprocessor
	postprocessors []interfaces.Postprocessor
}

// AudioProcessingOptions controls conservative, local-only preprocessing.
type AudioProcessingOptions struct {
	Normalize     bool
	TargetLUFS    float64
	ReduceNoise   bool
	TempDirectory string
}

// AudioQualityMetrics contains technical signal measurements only.
type AudioQualityMetrics struct {
	PeakDB        float64
	RMSDB         float64
	Clipped       bool
	TooQuiet      bool
	EmptyChannels []int
}

// NewProcessingPipeline creates a new processing pipeline
func NewProcessingPipeline() *ProcessingPipeline {
	pipeline := &ProcessingPipeline{
		preprocessors:  make([]interfaces.Preprocessor, 0),
		postprocessors: make([]interfaces.Postprocessor, 0),
	}

	// Register default preprocessors
	pipeline.RegisterPreprocessor(&AudioFormatPreprocessor{})

	return pipeline
}

// RegisterPreprocessor adds a preprocessor to the pipeline
func (p *ProcessingPipeline) RegisterPreprocessor(preprocessor interfaces.Preprocessor) {
	p.preprocessors = append(p.preprocessors, preprocessor)
}

// RegisterPostprocessor adds a postprocessor to the pipeline
func (p *ProcessingPipeline) RegisterPostprocessor(postprocessor interfaces.Postprocessor) {
	p.postprocessors = append(p.postprocessors, postprocessor)
}

// ProcessAudio applies all applicable preprocessors to the audio input
func (p *ProcessingPipeline) ProcessAudio(ctx context.Context, input interfaces.AudioInput, capabilities interfaces.ModelCapabilities, options AudioProcessingOptions) (interfaces.AudioInput, error) {
	currentInput := input

	for _, preprocessor := range p.preprocessors {
		if preprocessor.AppliesTo(capabilities) {
			logger.Info("Applying preprocessor", "type", fmt.Sprintf("%T", preprocessor))
			var processedInput interfaces.AudioInput
			var err error
			if audioPreprocessor, ok := preprocessor.(*AudioFormatPreprocessor); ok {
				processedInput, err = audioPreprocessor.ProcessWithOptions(ctx, currentInput, options)
			} else {
				processedInput, err = preprocessor.Process(ctx, currentInput)
			}
			if err != nil {
				return currentInput, err
			}
			currentInput = processedInput
		}
	}

	return currentInput, nil
}

// AudioFormatPreprocessor converts audio to required formats
type AudioFormatPreprocessor struct{}

// AppliesTo checks if this preprocessor should be used for the given model
func (a *AudioFormatPreprocessor) AppliesTo(capabilities interfaces.ModelCapabilities) bool {
	// Apply to all models for consistent audio format (mono 16kHz)
	return true
}

// GetRequiredFormats returns the output formats this preprocessor can produce
func (a *AudioFormatPreprocessor) GetRequiredFormats() []string {
	return []string{"wav"}
}

// Process converts audio to the required format
func (a *AudioFormatPreprocessor) Process(ctx context.Context, input interfaces.AudioInput) (interfaces.AudioInput, error) {
	return a.ProcessWithOptions(ctx, input, AudioProcessingOptions{})
}

// ProcessWithOptions validates signal quality and creates a derived, model-ready file.
func (a *AudioFormatPreprocessor) ProcessWithOptions(ctx context.Context, input interfaces.AudioInput, options AudioProcessingOptions) (interfaces.AudioInput, error) {
	// Check if conversion is needed
	requiredFormat := "wav"
	requiredSampleRate := 16000
	requiredChannels := 1

	metrics, analysisErr := analyzeAudioQuality(ctx, input.FilePath, input.Channels)
	if analysisErr != nil {
		logger.Warn("Audio quality analysis unavailable", "error", analysisErr)
	} else {
		logger.Info("Audio quality analysis",
			"peak_db", metrics.PeakDB,
			"rms_db", metrics.RMSDB,
			"clipped", metrics.Clipped,
			"too_quiet", metrics.TooQuiet,
			"empty_channel_count", len(metrics.EmptyChannels))
		if len(metrics.EmptyChannels) > 0 {
			logger.Warn("Audio contains empty or near-empty channels", "channels", metrics.EmptyChannels)
		}
	}

	needsFiltering := options.Normalize || options.ReduceNoise
	if strings.ToLower(input.Format) == requiredFormat &&
		input.SampleRate == requiredSampleRate &&
		input.Channels == requiredChannels &&
		input.Metadata["probe_verified"] != "false" && !needsFiltering {
		// No conversion needed
		return input, nil
	}

	logger.Info("Converting audio format",
		"from_format", input.Format,
		"to_format", requiredFormat,
		"from_sample_rate", input.SampleRate,
		"to_sample_rate", requiredSampleRate,
		"from_channels", input.Channels,
		"to_channels", requiredChannels)

	// Create output path
	tempDirectory := options.TempDirectory
	if tempDirectory == "" {
		tempDirectory = filepath.Dir(input.FilePath)
	}
	if err := os.MkdirAll(tempDirectory, 0755); err != nil {
		return input, fmt.Errorf("failed to create audio preprocessing directory: %w", err)
	}
	outputFile, err := os.CreateTemp(tempDirectory, ".scriberr-audio-*.wav")
	if err != nil {
		return input, fmt.Errorf("failed to create derived audio file: %w", err)
	}
	outputPath := outputFile.Name()
	if err := outputFile.Close(); err != nil {
		_ = os.Remove(outputPath)
		return input, fmt.Errorf("failed to prepare derived audio file: %w", err)
	}

	// Build FFmpeg command
	args := []string{"-nostdin", "-i", input.FilePath}
	filters := buildAudioFilters(options)
	if len(filters) > 0 {
		args = append(args, "-af", strings.Join(filters, ","))
	}
	args = append(args,
		"-ar", strconv.Itoa(requiredSampleRate),
		"-ac", strconv.Itoa(requiredChannels),
		"-c:a", "pcm_s16le",
		"-y", // Overwrite output file
		outputPath,
	)

	// Execute FFmpeg
	cmd := exec.CommandContext(ctx, "ffmpeg", args...)
	_, err = cmd.CombinedOutput()
	if err != nil {
		_ = os.Remove(outputPath)
		logger.Error("FFmpeg audio preprocessing failed", "error", err)
		return input, fmt.Errorf("audio conversion failed: %w", err)
	}

	// Create new audio input
	convertedInput := interfaces.AudioInput{
		FilePath:     outputPath,
		Format:       requiredFormat,
		SampleRate:   requiredSampleRate,
		Channels:     requiredChannels,
		Duration:     input.Duration, // Preserve duration
		Size:         0,              // Will be set when file is read
		Metadata:     input.Metadata,
		TempFilePath: outputPath, // Mark as temporary
	}

	// Get file size
	if stat, err := os.Stat(outputPath); err == nil {
		convertedInput.Size = stat.Size()
	}

	logger.Info("Audio conversion completed", "output_size", convertedInput.Size)

	return convertedInput, nil
}

func buildAudioFilters(options AudioProcessingOptions) []string {
	filters := make([]string, 0, 2)
	if options.ReduceNoise {
		filters = append(filters, "afftdn=nr=8:nf=-35:tn=1")
	}
	if options.Normalize {
		target := options.TargetLUFS
		if target < -24 || target > -12 {
			target = -16
		}
		filters = append(filters, fmt.Sprintf("loudnorm=I=%.1f:TP=-1.5:LRA=11", target))
	}
	return filters
}

var (
	peakPattern = regexp.MustCompile(`(?m)Peak level dB:\s*(-?inf|[-+0-9.]+)`)
	rmsPattern  = regexp.MustCompile(`(?m)RMS level dB:\s*(-?inf|[-+0-9.]+)`)
)

func analyzeAudioQuality(ctx context.Context, path string, channels int) (AudioQualityMetrics, error) {
	cmd := exec.CommandContext(ctx, "ffmpeg", "-nostdin", "-hide_banner", "-i", path,
		"-af", "astats=metadata=0:reset=0", "-f", "null", "-")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return AudioQualityMetrics{}, fmt.Errorf("ffmpeg signal analysis failed: %w", err)
	}
	return parseAudioQualityMetrics(string(output), channels), nil
}

func parseAudioQualityMetrics(output string, channels int) AudioQualityMetrics {
	metrics := AudioQualityMetrics{PeakDB: -100, RMSDB: -100}
	peaks := parseDBValues(peakPattern, output)
	rms := parseDBValues(rmsPattern, output)
	if len(peaks) > 0 {
		metrics.PeakDB = peaks[len(peaks)-1]
	}
	if len(rms) > 0 {
		metrics.RMSDB = rms[len(rms)-1]
	}
	metrics.Clipped = metrics.PeakDB >= -0.1
	metrics.TooQuiet = metrics.RMSDB < -45
	for i := 0; i < channels && i < len(rms)-1; i++ { // final value is the overall channel
		if rms[i] <= -90 {
			metrics.EmptyChannels = append(metrics.EmptyChannels, i+1)
		}
	}
	return metrics
}

func parseDBValues(pattern *regexp.Regexp, output string) []float64 {
	var values []float64
	for _, match := range pattern.FindAllStringSubmatch(output, -1) {
		value := -100.0
		if !strings.EqualFold(match[1], "-inf") {
			parsed, err := strconv.ParseFloat(match[1], 64)
			if err != nil {
				continue
			}
			value = parsed
		}
		values = append(values, value)
	}
	return values
}

// VoiceActivityDetectionPreprocessor applies VAD preprocessing
type VoiceActivityDetectionPreprocessor struct{}

// AppliesTo checks if this preprocessor should be used
func (v *VoiceActivityDetectionPreprocessor) AppliesTo(capabilities interfaces.ModelCapabilities) bool {
	// Apply to models that benefit from VAD preprocessing
	return capabilities.Features["vad"]
}

// GetRequiredFormats returns the output formats this preprocessor can produce
func (v *VoiceActivityDetectionPreprocessor) GetRequiredFormats() []string {
	return []string{"wav", "mp3", "flac"}
}

// Process applies voice activity detection preprocessing
func (v *VoiceActivityDetectionPreprocessor) Process(ctx context.Context, input interfaces.AudioInput) (interfaces.AudioInput, error) {
	// For now, this is a placeholder
	// In a real implementation, this would apply VAD to remove silence
	logger.Info("VAD preprocessing (placeholder)", "file", input.FilePath)
	return input, nil
}

// NoiseReductionPreprocessor applies noise reduction
type NoiseReductionPreprocessor struct{}

// AppliesTo checks if this preprocessor should be used
func (n *NoiseReductionPreprocessor) AppliesTo(capabilities interfaces.ModelCapabilities) bool {
	// Apply to models that would benefit from noise reduction
	return capabilities.Features["high_quality"]
}

// GetRequiredFormats returns the output formats this preprocessor can produce
func (n *NoiseReductionPreprocessor) GetRequiredFormats() []string {
	return []string{"wav"}
}

// Process applies noise reduction preprocessing
func (n *NoiseReductionPreprocessor) Process(ctx context.Context, input interfaces.AudioInput) (interfaces.AudioInput, error) {
	// For now, this is a placeholder
	// In a real implementation, this would apply noise reduction using FFmpeg or other tools
	logger.Info("Noise reduction preprocessing (placeholder)", "file", input.FilePath)
	return input, nil
}

// TextPostprocessor handles transcription result post-processing
type TextPostprocessor struct{}

// ProcessTranscript processes transcription results
func (t *TextPostprocessor) ProcessTranscript(ctx context.Context, result *interfaces.TranscriptResult, params map[string]interface{}) (*interfaces.TranscriptResult, error) {
	// Apply text cleaning, formatting, etc.
	logger.Info("Post-processing transcript", "segments", len(result.Segments))

	// Example post-processing: trim whitespace from segments
	for i := range result.Segments {
		result.Segments[i].Text = strings.TrimSpace(result.Segments[i].Text)
	}

	return result, nil
}

// ProcessDiarization processes diarization results
func (t *TextPostprocessor) ProcessDiarization(ctx context.Context, result *interfaces.DiarizationResult, params map[string]interface{}) (*interfaces.DiarizationResult, error) {
	// Apply diarization result cleaning, speaker merging, etc.
	logger.Info("Post-processing diarization", "segments", len(result.Segments))
	return result, nil
}

// AppliesTo determines if this postprocessor should be used
func (t *TextPostprocessor) AppliesTo(capabilities interfaces.ModelCapabilities, params map[string]interface{}) bool {
	return true // Always apply text post-processing
}
