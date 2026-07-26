# Image Compression using the Discrete Cosine Transform (DCT)

This project implements DCT-based image compression **from scratch** — the DCT basis matrix, forward transform, coefficient truncation, and inverse transform are all built using plain NumPy linear algebra, without calling any built-in transform function (e.g. `scipy.fftpack.dct`).

This README explains the math behind each step.

---

## 1. The core idea

A natural image, viewed as a grid of pixel intensities, contains a lot of redundancy: nearby pixels tend to be similar, and most of the *visually important* information (overall shapes, smooth gradients) changes slowly across the image, while only a smaller portion (edges, fine texture) changes rapidly.

The DCT re-expresses an image not as pixel intensities, but as a sum of cosine waves of increasing frequency. Once in this form, the "slowly changing" information collapses into a handful of low-frequency coefficients, and the "rapidly changing" detail is spread across many high-frequency coefficients that are individually small. Discarding those small, high-frequency coefficients loses relatively little visual quality while dropping most of the actual numbers stored — this is the basis of compression, and is conceptually the same principle JPEG uses.

---

## 2. Building the DCT basis matrix

The 2D DCT-II basis matrix $C$ is defined element-wise as:

$$
C[m, n] = \alpha(m) \cdot \cos\left( \frac{\pi (2n + 1) m}{2N} \right)
$$

for $m, n = 0, 1, \dots, N-1$, where $N$ is the image's side length (e.g. 512), and the normalization constant $\alpha(m)$ is:

$$
\alpha(m) =
\begin{cases}
\sqrt{\dfrac{1}{N}}, & m = 0 \\
\sqrt{\dfrac{2}{N}}, & m \neq 0
\end{cases}
$$

**Interpretation of each row:**
- Row $m = 0$ is a constant vector — this is the **DC term**, representing the average brightness of the signal.
- Rows $m > 0$ are cosine waves of increasing frequency — these are the **AC terms**, representing detail and edges at progressively finer scales.

Each row of $C$ is one such basis vector, and together they form a complete set of building blocks: any signal of length $N$ can be written as a weighted sum of these $N$ cosine waves.

---

## 3. Why orthonormality matters

$C$ is constructed so that its rows are both:
- **Orthogonal** — any two different basis vectors are perpendicular (their dot product is 0), meaning they capture independent, non-redundant information.
- **Normalized** — each basis vector has unit length (magnitude 1), enforced by the $\alpha(m)$ scaling factor above.

Together, this makes $C$ an **orthonormal matrix**, satisfying:

$$
C^\top C = I
$$

This property is verified directly in the notebook by checking `C.T @ C` against the identity matrix.

**Why this matters:** orthonormality is exactly what makes the transform *reversible*. Applying $C$ forward and $C^\top$ backward reconstructs the original signal exactly (assuming no coefficients were discarded) — without this property, there would be no guarantee that transforming and inverse-transforming gets you back to where you started.

---

## 4. Forward transform: pixels → frequency domain

Each color channel (Red, Green, Blue) is a 2D array of pixel intensities. The 2D DCT is applied by multiplying by $C$ on both sides:

$$
D = C^\top \ A \ C
$$

where $A$ is the original channel and $D$ is the resulting matrix of DCT coefficients, the same shape as $A$, but now representing *frequency content* rather than *pixel intensity*.

Applied to each channel independently:

$$
D_R = C^\top R \ C \qquad D_G = C^\top G \ C \qquad, D_B = C^\top B \ C
$$

**Why $C^\top$ on the left and $C$ on the right?** The image is a 2D signal, so the transform needs to be applied along both dimensions — rows and columns. Multiplying by $C^\top$ on the left transforms the columns; multiplying by $C$ on the right transforms the rows. The result is a full 2D frequency decomposition.

**Where the frequency information ends up:** in $D$, low-frequency content (broad shapes, smooth gradients) concentrates near the top-left corner ($D[0,0]$ being the pure DC/average-brightness term), while high-frequency content (sharp edges, fine texture) appears toward the bottom-right.

---

## 5. Compression: truncating the coefficients

This is the actual lossy compression step. Given a chosen $K \le N$, only the top-left $K \times K$ block of each channel's coefficient matrix is kept; everything else is set to zero:

$$
D_{\text{compressed}}[m, n] =
\begin{cases}
D[m, n], & m < K \text{ and } n < K \\
0, & \text{otherwise}
\end{cases}
$$

Since most of a natural image's energy is concentrated in the low-frequency (top-left) region, zeroing out the remaining high-frequency coefficients discards relatively little visually important information — the smaller $K$ is, the more aggressive the compression, and the blurrier the eventual reconstruction.

Note that $D_{\text{compressed}}$ is still an $N \times N$ matrix in this implementation (the zeroed entries are still stored as explicit zeros) — a real compressor would only store the nonzero $K \times K$ block itself (that's where the actual file-size reduction comes from, at a ratio of $K^2 / N^2$), and reconstruct the full $N \times N$ matrix only at decode time.

---

## 6. Inverse transform: frequency domain → pixels

Because $C$ is orthonormal, the transform is invertible by applying $C$ and $C^\top$ from the opposite sides:

$$
A_{\text{compressed}} = C \, D_{\text{compressed}} \, C^\top
$$

Applied per channel:

$$
R_{\text{compressed}} = C \, D_{R,\text{compressed}} \, C^\top, \qquad
G_{\text{compressed}} = C \, D_{G,\text{compressed}} \, C^\top, \qquad
B_{\text{compressed}} = C \, D_{B,\text{compressed}} \, C^\top
$$

Because $D_{\text{compressed}}$ had its high-frequency entries zeroed out, this reconstruction is only an **approximation** of the original channel, not an exact copy — the smaller $K$ was, the more information was discarded, and the more the reconstruction visibly deviates from the original (more blur, loss of fine detail).

---

## 7. Clipping back to valid pixel values

The inverse transform produces plain floating-point numbers, which can legitimately overshoot slightly outside the valid intensity range (e.g. producing $-2.3$ or $259.7$) due to reconstruction error from the discarded coefficients. Since the image is stored using 8 bits per channel (`uint8`), valid pixel values must be integers in $[0, 255]$ — that upper bound comes from $2^8 - 1 = 255$, fixed by the bit depth, and has nothing to do with the image's size ($N$) or how many coefficients were kept ($K$). Each channel is clipped into this range *before* casting to `uint8`, otherwise out-of-range values would wrap around unpredictably instead of saturating at black/white.

---

## Summary of the pipeline

```
A (pixel domain)
   │  D = Cᵀ A C           (forward DCT)
   ▼
D (frequency domain, N×N)
   │  keep only top-left K×K, zero the rest      (compression)
   ▼
D_compressed (frequency domain, N×N, mostly zeros)
   │  A_compressed = C D_compressed Cᵀ    (inverse DCT)
   ▼
A_compressed (pixel domain, approximation)
   │  clip to [0, 255], cast to uint8
   ▼
final compressed image
```
