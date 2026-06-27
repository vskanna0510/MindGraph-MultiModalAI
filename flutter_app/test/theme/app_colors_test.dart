import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mindgraph_plus_plus/core/design_system/tokens/app_colors.dart';

void main() {
  test('light palette matches HDL v3 spec', () {
    expect(AppColors.primary, const Color(0xFF5E9ED6));
    expect(AppColors.secondary, const Color(0xFF4CAF9C));
    expect(AppColors.accent, const Color(0xFF7C8CF8));
    expect(AppColors.success, const Color(0xFF34D399));
    expect(AppColors.danger, const Color(0xFFEF4444));
    expect(AppColors.background, const Color(0xFFF8FAFC));
  });

  test('dark palette matches HDL v3 spec', () {
    expect(AppColors.darkBackground, const Color(0xFF09090B));
    expect(AppColors.darkSurface, const Color(0xFF18181B));
    expect(AppColors.darkCard, const Color(0xFF202024));
    expect(AppColors.darkPrimary, const Color(0xFF7CB5F0));
    expect(AppColors.darkSecondary, const Color(0xFF5AD4B8));
  });
}
