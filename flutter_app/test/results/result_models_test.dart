import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plusplus/features/results/data/result_analysis_builder.dart';
import 'package:mindgraph_plusplus/features/results/domain/result_models.dart';
import 'package:mindgraph_plusplus/features/record/domain/recording_models.dart';

void main() {
  group('ResultAnalysisBuilder', () {
    test('builds non-diagnostic narrative', () {
      final result = ResultAnalysisBuilder().build(
        sessionId: 'test_1',
        mode: RecordingMode.combined,
        durationSeconds: 90,
        modalities: ['audio', 'text'],
        processedLocally: true,
        isOffline: false,
      );
      expect(result.riskNarrative, isNot(contains('depressed')));
      expect(result.insights.length, 3);
      expect(result.modalityContributions, isNotEmpty);
    });

    test('shows support resources when elevated', () {
      final builder = ResultAnalysisBuilder();
      SessionAnalysisResult? elevated;
      for (var i = 0; i < 50; i++) {
        final r = builder.build(
          sessionId: 'elevated_$i',
          mode: RecordingMode.voice,
          durationSeconds: 60,
          modalities: ['audio'],
          processedLocally: true,
          isOffline: false,
        );
        if (r.isElevated) {
          elevated = r;
          break;
        }
      }
      if (elevated != null) {
        expect(elevated.showSupportResources, isTrue);
      }
    });
  });

  group('SessionAnalysisResult', () {
    test('json roundtrip', () {
      final result = ResultAnalysisBuilder().build(
        sessionId: 'roundtrip',
        mode: RecordingMode.text,
        durationSeconds: 45,
        modalities: ['text'],
        processedLocally: true,
        isOffline: true,
      );
      final restored = SessionAnalysisResult.fromJson(result.toJson());
      expect(restored.sessionId, 'roundtrip');
      expect(restored.featureContributions.length, result.featureContributions.length);
    });
  });

  group('AiPipelineProgress', () {
    test('copyWith updates progress', () {
      const p = AiPipelineProgress(overallProgress: 0.5);
      expect(p.copyWith(overallProgress: 0.8).overallProgress, 0.8);
    });
  });
}
