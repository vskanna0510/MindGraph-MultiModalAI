import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plusplus/features/graph/domain/graph_models.dart';

void main() {
  group('GraphSnapshot', () {
    test('copyWith timeline index', () {
      const snap = GraphSnapshot(nodes: [], edges: [], timelineIndex: 0, timelineLength: 3);
      expect(snap.copyWith(timelineIndex: 2).timelineIndex, 2);
    });
  });

  group('GraphNode', () {
    test('toggle expanded', () {
      const node = GraphNode(
        id: 'n1',
        label: 'Calm',
        type: GraphNodeType.emotion,
        x: 0.5,
        y: 0.5,
      );
      expect(node.copyWith(expanded: false).expanded, isFalse);
    });
  });
}
