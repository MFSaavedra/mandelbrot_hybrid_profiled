/**
 * @file mandelframe.h
 * @brief Per-frame image buffer, camera geometry, and the atomic completion
 *        counter that decides when a frame is fully rendered.
 *
 * One @ref MandelFrame exists per output image in the interpolated zoom
 * sequence. It owns the @c QImage pixel buffer and the complex-plane rectangle
 * the frame maps onto, and it tracks how many regions are still outstanding so
 * the worker that finishes the last one can save the PNG.
 */
#ifndef MANDELFRAME_H
#define MANDELFRAME_H

#include <QImage>
#include <QAtomicInt>
#include <QVector>
#include <QMutex>

/// Maximum length of an image-file prefix / frame filename (buffer sizing).
const int MAXFNAME=50;

/**
 * @brief One subdivision node recorded during a @c vizMode run.
 *
 * Every examined region — both internal split-nodes and computed leaves —
 * registers its pixel rectangle, recursion depth, whether it became a leaf,
 * and (for leaves) which executor computed it. @ref main.cpp replays the
 * recorded rectangles to draw the load-balancing partition and the
 * depth-by-depth subdivision animation. Recorded only when @c vizMode is set,
 * so it never touches a timed run.
 */
struct VizRect
{
  int  x, y, w, h;   ///< Pixel rectangle within the frame.
  int  depth;        ///< Recursion depth (root = 0).
  bool leaf;         ///< @c true if computed, @c false if split into four children.
  bool onGPU;        ///< Executor: @c true = GPU thread (meaningful only when @c leaf).
};

/**
 * @brief A single output image frame: pixel buffer, complex-plane mapping, and
 *        atomic region-completion counter.
 *
 * The complex rectangle [@c upperX,@c upperY]&ndash;[@c lowerX,@c lowerY] maps
 * onto a @c pixelsX &times; @c pixelsY @c QImage; @c stepX / @c stepY are the
 * per-pixel complex increments. Frame completion is tracked by the atomic @ref
 * remainingRegions using a net-count scheme (see @ref regionSplit / @ref
 * regionComplete): when it reaches zero the frame is saved to disk.
 */
class MandelFrame
{
public:
    int MAXITER;                ///< Iteration ceiling for every pixel in this frame.
    int frameIndex;             ///< Position in the interpolated sequence (global index).
    double upperX, upperY, lowerX, lowerY;  ///< Complex-plane rectangle (upper-left / lower-right).
    double stepX, stepY;        ///< Per-pixel complex increment in X and Y.
    int pixelsX, pixelsY;       ///< Frame resolution in pixels.
    QImage *img;                ///< The output pixel buffer.
    char fname[MAXFNAME+1];     ///< Output PNG filename (carries the global frame index).
    /// Net outstanding-region counter. Starts at 1 (the root region); a split
    /// adds 3, a completion subtracts 1; hitting 0 means the frame is done.
    QAtomicInt remainingRegions;

    /// Recorded subdivision nodes (populated only when @c vizMode is set).
    /// Guarded by @ref vizLock because every worker appends concurrently from
    /// @ref MandelRegion::examine.
    QVector<VizRect> vizRects;
    QMutex vizLock;             ///< Serializes concurrent @ref addVizRect appends.

    /**
     * @brief Construct a frame over a complex rectangle at a given resolution.
     * @param uX,uY   Upper-left corner in the complex plane.
     * @param lX,lY   Lower-right corner in the complex plane.
     * @param pX,pY   Frame resolution in pixels.
     * @param c       Output filename (copied into @ref fname).
     * @param maxiter Iteration ceiling (@ref MAXITER).
     */
    MandelFrame(double, double, double, double, int, int, char *, int );
    /** @brief Account for a four-way split: net +3 to @ref remainingRegions. */
    void regionSplit();
    /** @brief Account for a finished leaf: -1 to @ref remainingRegions; saves the
     *         PNG when the count reaches zero. */
    void regionComplete();
    /** @brief Thread-safe append of one subdivision node (viz path only). */
    void addVizRect(const VizRect &r);
};

#endif