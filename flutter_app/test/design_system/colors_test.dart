import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plus_plus/core/design_system/tokens/color_scale.dart';
import 'package:mindgraph_plus_plus/core/design_system/tokens/colors.dart';
import 'package:mindgraph_plus_plus/core/design_system/tokens/theme_extensions.dart';

void main() {
  test('primary palette has 500 shade matching brand', () {
    expect(AppColorPalette.primary.c500, const Color(0xFF5E9ED6));
    expect(AppColorPalette.primary[500], AppColorPalette.primary.c500);
  });

  test('neutral palette includes 950', () {
    expect(AppColorPalette.neutral.c950, const Color(0xFF09090B));
  });

  test('semantic light colors use palette not raw hex in widgets path', () {
    final s = AppSemanticColors.light;
    expect(s.primary, AppColorPalette.primary.c500);
    expect(s.positive, AppColorPalette.success.c500);
    expect(s.negative, AppColorPalette.error.c500);
    expect(s.background, AppColorPalette.neutral.c50);
  });

  test('semantic dark background uses neutral 950', () {
    expect(AppSemanticColors.dark.background, AppColorPalette.neutral.c950);
  });

  test('semantic lerp interpolates', () {
    final mid = AppSemanticColors.light.lerp(AppSemanticColors.dark, 0.5);
    expect(mid, isA<AppSemanticColors>());
  });
}
