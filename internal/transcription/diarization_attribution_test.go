package transcription

import (
	"testing"

	"scriberr/internal/transcription/interfaces"
)

func TestMergeDiarizationWithTranscriptionUsesExclusiveWordOverlap(t *testing.T) {
	service := &UnifiedTranscriptionService{}
	speakerA := "SPEAKER_00"
	speakerB := "SPEAKER_01"

	transcript := &interfaces.TranscriptResult{
		Segments: []interfaces.TranscriptSegment{
			{Start: 0.0, End: 1.2, Text: "hello yes continue"},
		},
		WordSegments: []interfaces.TranscriptWord{
			{Start: 0.00, End: 0.40, Word: "hello"},
			{Start: 0.45, End: 0.62, Word: "yes"},
			{Start: 0.70, End: 1.20, Word: "continue"},
		},
	}
	diarization := &interfaces.DiarizationResult{
		Segments: []interfaces.DiarizationSegment{
			{Start: 0.0, End: 1.2, Speaker: speakerA, Confidence: 1.0},
		},
		ExclusiveSegments: []interfaces.DiarizationSegment{
			{Start: 0.0, End: 0.44, Speaker: speakerA, Confidence: 1.0},
			{Start: 0.44, End: 0.66, Speaker: speakerB, Confidence: 1.0},
			{Start: 0.66, End: 1.2, Speaker: speakerA, Confidence: 1.0},
		},
	}

	merged := service.mergeDiarizationWithTranscription(transcript, diarization)

	if got := len(merged.Segments); got != 3 {
		t.Fatalf("expected 3 speaker turns, got %d: %#v", got, merged.Segments)
	}
	if got := speakerValue(merged.WordSegments[1].Speaker); got != speakerB {
		t.Fatalf("expected short word to keep speaker B, got %q", got)
	}
	if got := speakerValue(merged.Segments[1].Speaker); got != speakerB {
		t.Fatalf("expected middle segment speaker B, got %q", got)
	}
	if got := merged.Metadata["speaker_attribution_timeline"]; got != "exclusive_speaker_diarization" {
		t.Fatalf("expected exclusive timeline metadata, got %q", got)
	}
}

func TestApplySpeakerContinuityOnlyFixesTinyFragments(t *testing.T) {
	service := &UnifiedTranscriptionService{}
	speakerA := "SPEAKER_00"
	speakerB := "SPEAKER_01"

	segments := []interfaces.TranscriptSegment{
		{Start: 0.0, End: 1.0, Text: "I wanted", Speaker: &speakerA},
		{Start: 1.05, End: 1.20, Text: "uh", Speaker: &speakerB},
		{Start: 1.25, End: 2.0, Text: "to ask", Speaker: &speakerA},
		{Start: 2.4, End: 3.0, Text: "yes absolutely", Speaker: &speakerB},
		{Start: 3.1, End: 4.0, Text: "thanks", Speaker: &speakerA},
	}

	processed := service.applySpeakerContinuity(segments)

	if got := speakerValue(processed[1].Speaker); got != speakerA {
		t.Fatalf("expected tiny filler fragment to be smoothed to A, got %q", got)
	}
	if got := speakerValue(processed[3].Speaker); got != speakerB {
		t.Fatalf("expected real short intervention to remain B, got %q", got)
	}
}
