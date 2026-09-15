package com.swift.guard

import android.accessibilityservice.AccessibilityService
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo

class AutoMergeAccessibilityService : AccessibilityService() {

    companion object {
        var isAutoMergeEnabled = true
        private const val TAG = "SWIFT_AutoMerge"

        // Common labels across Samsung, Google Phone, OnePlus/Realme ColorOS, Xiaomi MIUI dialers
        private val MERGE_TARGETS = listOf(
            "merge",
            "merge calls",
            "merge call",
            "combine",
            "conference",
            "join"
        )
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        if (!isAutoMergeEnabled || event == null) return

        // Package filter: Only react when inside in-call or phone dialer apps!
        val pkg = event.packageName?.toString()?.lowercase() ?: ""
        val isDialerApp = pkg.contains("dialer") ||
                pkg.contains("incall") ||
                pkg.contains("telecom") ||
                pkg.contains("phone") ||
                pkg.contains("calling") ||
                pkg.contains("com.android.server.telecom") ||
                pkg.contains("google.android.dialer") ||
                pkg.contains("samsung.android.incallui") ||
                pkg.contains("oplus") && pkg.contains("phone")

        if (!isDialerApp) {
            return
        }

        val rootNode = rootInActiveWindow ?: return
        try {
            searchAndClickMerge(rootNode)
        } catch (e: Exception) {
            Log.e(TAG, "Error evaluating node: ${e.message}")
        }
    }

    private fun searchAndClickMerge(node: AccessibilityNodeInfo): Boolean {
        // 1. Check text content
        val text = node.text?.toString()?.trim()?.lowercase()
        val desc = node.contentDescription?.toString()?.trim()?.lowercase()

        for (target in MERGE_TARGETS) {
            val matchesText = text != null && (text == target || text.contains(target))
            val matchesDesc = desc != null && (desc == target || desc.contains(target))

            if (matchesText || matchesDesc) {
                if (performClickOnNodeOrParent(node)) {
                    Log.d(TAG, "Successfully auto-clicked Merge Calls! Target: $target")
                    return true
                }
            }
        }

        // 2. Recursively traverse child nodes in the in-call dialer view hierarchy
        for (i in 0 until node.childCount) {
            val child = node.getChild(i)
            if (child != null) {
                val clicked = searchAndClickMerge(child)
                if (clicked) return true
            }
        }

        return false
    }

    private fun performClickOnNodeOrParent(node: AccessibilityNodeInfo): Boolean {
        var current: AccessibilityNodeInfo? = node
        while (current != null) {
            if (current.isClickable) {
                val success = current.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                if (success) return true
            }
            current = current.parent
        }
        return false
    }

    override fun onInterrupt() {
        Log.w(TAG, "Accessibility Service Interrupted")
    }
}
