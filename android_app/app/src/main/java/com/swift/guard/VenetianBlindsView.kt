package com.swift.guard

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

/**
 * Venetian Blinds overlay view.
 * Draws N horizontal slats that each reveal from 0→full height sequentially,
 * producing a classic Venetian-blinds wipe effect.
 *
 * Usage: add to layout at full screen, call animate(onDone) to run.
 */
class VenetianBlindsView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val SLAT_COUNT = 10
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private var progress = 0f   // 0.0 → 1.0

    // Each slat covers (height / SLAT_COUNT) pixels, revealing top-to-bottom
    override fun onDraw(canvas: Canvas) {
        val slatH = height.toFloat() / SLAT_COUNT
        for (i in 0 until SLAT_COUNT) {
            // stagger: slat i starts at progress = i/SLAT_COUNT
            val slatProgress = ((progress * SLAT_COUNT) - i).coerceIn(0f, 1f)
            val top = i * slatH
            val bottom = top + slatH * slatProgress
            if (bottom > top) {
                canvas.drawRect(0f, top, width.toFloat(), bottom, paint)
            }
        }
    }

    fun setColor(color: Int) {
        paint.color = color
        invalidate()
    }

    fun animate(durationMs: Long = 500L, onDone: () -> Unit) {
        val animator = android.animation.ValueAnimator.ofFloat(0f, 1f).apply {
            duration = durationMs
            interpolator = android.view.animation.AccelerateDecelerateInterpolator()
            addUpdateListener {
                progress = it.animatedValue as Float
                invalidate()
            }
            addListener(object : android.animation.AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: android.animation.Animator) {
                    onDone()
                }
            })
        }
        animator.start()
    }

    fun reverseAnimate(durationMs: Long = 400L, onDone: () -> Unit) {
        val animator = android.animation.ValueAnimator.ofFloat(1f, 0f).apply {
            duration = durationMs
            interpolator = android.view.animation.AccelerateInterpolator()
            addUpdateListener {
                progress = it.animatedValue as Float
                invalidate()
            }
            addListener(object : android.animation.AnimatorListenerAdapter() {
                override fun onAnimationEnd(animation: android.animation.Animator) {
                    onDone()
                }
            })
        }
        animator.start()
    }
}
