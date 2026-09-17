package pipeline

import (
	"reflect"
	"testing"
)

func TestParseAudioQualityMetrics(t *testing.T) {
	output := `[Parsed_astats_0]
Channel: 1
Peak level dB: -0.050000
RMS level dB: -22.000000
Channel: 2
Peak level dB: -100.000000
RMS level dB: -inf
Overall
Peak level dB: -0.050000
RMS level dB: -24.000000`

	metrics := parseAudioQualityMetrics(output, 2)
	if !metrics.Clipped {
		t.Fatal("expected clipping warning")
	}
	if metrics.TooQuiet {
		t.Fatal("did not expect overall signal to be considered too quiet")
	}
	if !reflect.DeepEqual(metrics.EmptyChannels, []int{2}) {
		t.Fatalf("unexpected empty channels: %v", metrics.EmptyChannels)
	}
}

func TestBuildAudioFiltersIsConservativeByDefault(t *testing.T) {
	if filters := buildAudioFilters(AudioProcessingOptions{}); len(filters) != 0 {
		t.Fatalf("default preprocessing must not alter signal: %v", filters)
	}

	filters := buildAudioFilters(AudioProcessingOptions{
		Normalize:   true,
		TargetLUFS:  -16,
		ReduceNoise: true,
	})
	want := []string{"afftdn=nr=8:nf=-35:tn=1", "loudnorm=I=-16.0:TP=-1.5:LRA=11"}
	if !reflect.DeepEqual(filters, want) {
		t.Fatalf("unexpected filters: got %v want %v", filters, want)
	}
}

func TestBuildAudioFiltersBoundsTarget(t *testing.T) {
	filters := buildAudioFilters(AudioProcessingOptions{Normalize: true, TargetLUFS: 0})
	want := []string{"loudnorm=I=-16.0:TP=-1.5:LRA=11"}
	if !reflect.DeepEqual(filters, want) {
		t.Fatalf("unsafe target should fall back to -16 LUFS: %v", filters)
	}
}
