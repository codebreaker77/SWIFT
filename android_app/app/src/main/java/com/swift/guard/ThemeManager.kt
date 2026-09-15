package com.swift.guard

import android.content.Context

object ThemeManager {
    private const val PREFS = "swift_prefs"
    private const val KEY_DARK = "dark_mode"

    fun isDark(context: Context): Boolean =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .getBoolean(KEY_DARK, false)

    fun setDark(context: Context, dark: Boolean) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit().putBoolean(KEY_DARK, dark).apply()
    }

    fun toggle(context: Context): Boolean {
        val next = !isDark(context)
        setDark(context, next)
        return next
    }

    // Color tokens
    fun bg(context: Context): Int = if (isDark(context)) 0xFF0D0D0D.toInt() else 0xFFFFFFFF.toInt()
    fun surface(context: Context): Int = if (isDark(context)) 0xFF1A1A1A.toInt() else 0xFFF7F7F7.toInt()
    fun primary(context: Context): Int = if (isDark(context)) 0xFFFFFFFF.toInt() else 0xFF000000.toInt()
    fun secondary(context: Context): Int = if (isDark(context)) 0xFF888888.toInt() else 0xFF888888.toInt()
    fun divider(context: Context): Int = if (isDark(context)) 0xFF2A2A2A.toInt() else 0xFFE8E8E8.toInt()
    fun accent(context: Context): Int = if (isDark(context)) 0xFFFFFFFF.toInt() else 0xFF000000.toInt()
}
