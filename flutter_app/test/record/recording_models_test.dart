import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plusplus/features/record/domain/recording_models.dart';

void main() {
  group('RecordingState', () {
    test('canFinish when elapsed meets minimum', () {
      const state = RecordingState(elapsedSeconds: 30, minDurationSeconds: 30);
      expect(state.canFinish, isTrue);
    });

    test('isRecording during active recording', () {
      const state = RecordingState(buttonState: RecordButtonState.recording);
      expect(state.isRecording, isTrue);
    });
  });

  group('JournalDraft', () {
    test('word and char counts', () {
      const draft = JournalDraft(text: 'Hello world today');
      expect(draft.wordCount, 3);
      expect(draft.charCount, 17);
    });

    test('json roundtrip', () {
      const draft = JournalDraft(text: 'Test entry', emotionTags: ['calm']);
      final restored = JournalDraft.fromJson(draft.toJson());
      expect(restored.text, 'Test entry');
      expect(restored.emotionTags, ['calm']);
    });
  });

  group('QualitySignals', () {
    test('gentleHint for low lighting', () {
      const signals = QualitySignals(lighting: 0.3, faceVisibility: 0.8, audioQuality: 0.8);
      expect(signals.gentleHint, QualityHint.improveLighting);
    });
  });

  group('RecordingSessionResult', () {
    test('json roundtrip', () {
      const result = RecordingSessionResult(
        sessionId: 'abc',
        mode: RecordingMode.combined,
        durationSeconds: 90,
        modalities: ['audio', 'video'],
      );
      final restored = RecordingSessionResult.fromJson(result.toJson());
      expect(restored.sessionId, 'abc');
      expect(restored.modalities.length, 2);
    });
  });
}
