/* Numerical kernels for the existing quasi-static Stark profile algorithm.
 * Atomic physics, field probabilities, grid selection, and FFT convolution
 * remain in Python.  Use the buffer protocol, not NumPy's C API.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include "humlicek_w4.h"

static int
double_array(PyObject *object, Py_buffer *view, int writable)
{
    if (PyObject_GetBuffer(object, view,
            PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES |
            (writable ? PyBUF_WRITABLE : 0)) < 0) return 0;
    if (view->itemsize != sizeof(double) || view->format == NULL ||
        !(view->format[0] == 'd' ||
          ((view->format[0] == '@' || view->format[0] == '=') &&
           view->format[1] == 'd'))) {
        PyErr_SetString(PyExc_TypeError, "Stark arrays must have native float64 dtype");
        return 0;
    }
    if (!PyBuffer_IsContiguous(view, 'C')) {
        PyErr_SetString(PyExc_ValueError, "Stark arrays must be C-contiguous");
        return 0;
    }
    return 1;
}

/* Output may not overwrite inputs that a later profile sample still needs. */
static int
arrays_overlap(const Py_buffer *a, const Py_buffer *b)
{
    uintptr_t start_a = (uintptr_t)a->buf;
    uintptr_t start_b = (uintptr_t)b->buf;
    if (a->len == 0 || b->len == 0) return 0;
    if (start_a <= start_b) return start_b - start_a < (uintptr_t)a->len;
    return start_a - start_b < (uintptr_t)b->len;
}

typedef struct {
    double center_frequency, per_angstrom, scale, damping;
} FrequencyVoigt;

static void
frequency_voigt_setup(FrequencyVoigt *profile, double center,
                      double sigma, double gamma)
{
    const double light_speed = 2.99792458e10;
    const double center_cm = center * 1.0e-8;
    const double per_angstrom = light_speed / (center_cm * center_cm) * 1.0e-8;
    const double sigma_nu = fmax(sigma, 1.0e-12) * per_angstrom;
    const double gamma_nu = fmax(gamma, 0.0) * per_angstrom;
    profile->center_frequency = light_speed / center_cm;
    profile->per_angstrom = per_angstrom;
    profile->scale = 1.0 / (sigma_nu * sqrt(2.0));
    profile->damping = gamma_nu * profile->scale;
}

static double
frequency_voigt_value(const FrequencyVoigt *profile, double wavelength)
{
    const double light_speed = 2.99792458e10;
    const double pi = 3.1415926535897932384626433832795;
    const double detuning = light_speed / (wavelength * 1.0e-8)
        - profile->center_frequency;
    /* Match the released operation order and line-centre conversion.  The
     * frequency coordinate supplies wing asymmetry; there is no additional
     * wavelength-dependent conversion factor. */
    return openwd_humlicek_w4_real(detuning * profile->scale, profile->damping)
        * profile->scale / sqrt(pi) * profile->per_angstrom;
}

PyObject *
openwd_frequency_voigt_profile(PyObject *self, PyObject *args)
{
    PyObject *wave_object, *out_object, *result = NULL;
    Py_buffer wave = {0}, out = {0};
    double center, sigma, gamma;
    FrequencyVoigt profile;
    (void)self;
    if (!PyArg_ParseTuple(args, "OdddO:frequency_voigt_profile", &wave_object,
                          &center, &sigma, &gamma, &out_object)) return NULL;
    if (!double_array(wave_object, &wave, 0) ||
        !double_array(out_object, &out, 1)) goto done;
    if (wave.ndim != 1 || out.ndim != 1 || wave.shape[0] != out.shape[0] ||
        !isfinite(center) || center <= 0.0 ||
        !isfinite(sigma) || !isfinite(gamma)) {
        PyErr_SetString(PyExc_ValueError, "invalid frequency-Voigt arrays or widths");
        goto done;
    }
    if (arrays_overlap(&wave, &out)) {
        PyErr_SetString(PyExc_ValueError, "frequency-Voigt output must not overlap input");
        goto done;
    }
    frequency_voigt_setup(&profile, center, sigma, gamma);
    {
        const double *x = wave.buf;
        double *y = out.buf;
        Py_ssize_t i;
        Py_BEGIN_ALLOW_THREADS
        for (i = 0; i < wave.shape[0]; ++i) {
            y[i] = frequency_voigt_value(&profile, x[i]);
        }
        Py_END_ALLOW_THREADS
    }
    Py_INCREF(Py_None); result = Py_None;
done:
    if (wave.obj) PyBuffer_Release(&wave);
    if (out.obj) PyBuffer_Release(&out);
    return result;
}

PyObject *
openwd_pseudo_voigt_profile(PyObject *self, PyObject *args)
{
    PyObject *wave_object, *out_object, *result = NULL;
    Py_buffer wave = {0}, out = {0};
    double center, sigma, gamma;
    const double pi = 3.1415926535897932384626433832795;
    const double ln2 = 0.69314718055994530941723212145818;
    (void)self;
    if (!PyArg_ParseTuple(args, "OdddO:pseudo_voigt_profile", &wave_object,
                          &center, &sigma, &gamma, &out_object)) return NULL;
    if (!double_array(wave_object, &wave, 0) ||
        !double_array(out_object, &out, 1)) goto done;
    if (wave.ndim != 1 || out.ndim != 1 || wave.shape[0] != out.shape[0] ||
        !isfinite(center) || !isfinite(sigma) || !isfinite(gamma)) {
        PyErr_SetString(PyExc_ValueError, "invalid pseudo-Voigt arrays or widths");
        goto done;
    }
    {
        double g = 2.0 * sqrt(2.0 * ln2) * fmax(sigma, 1.0e-12);
        double l = 2.0 * fmax(gamma, 0.0);
        double width = pow(pow(g, 5.0) + 2.69269*pow(g, 4.0)*l +
            2.42843*pow(g, 3.0)*pow(l, 2.0) +
            4.47163*pow(g, 2.0)*pow(l, 3.0) +
            0.07842*g*pow(l, 4.0) + pow(l, 5.0), 0.2);
        double r = l / width;
        double mix = fmin(1.0, fmax(0.0, 1.36603*r - 0.47719*r*r + 0.11116*r*r*r));
        const double *x = wave.buf;
        double *y = out.buf;
        Py_ssize_t i;
        Py_BEGIN_ALLOW_THREADS
        for (i = 0; i < wave.shape[0]; ++i) {
            double z = (x[i] - center) / width;
            double gaussian = 2.0*sqrt(ln2)/(sqrt(pi)*width) * exp(-4.0*ln2*z*z);
            double lorentz = 2.0/(pi*width)/(1.0 + 4.0*z*z);
            y[i] = (1.0 - mix)*gaussian + mix*lorentz;
        }
        Py_END_ALLOW_THREADS
    }
    Py_INCREF(Py_None);
    result = Py_None;
done:
    if (wave.obj) PyBuffer_Release(&wave);
    if (out.obj) PyBuffer_Release(&out);
    return result;
}

/* Deposit a uniform box, using the same piecewise-linear CDF as the Python
 * reference.  Integrating the deposited second difference once gives bin
 * masses: the reference's second cumsum followed by diff cancels algebraically.
 * Clip the extent, not its original slope, to discard out-of-window mass.
 */
static void
deposit_box(double low, double high, double mass, double step,
            Py_ssize_t size, double *second)
{
    double origin = -((size - 1) / 2 + 0.5) * step;
    double a = (low - origin) / step;
    double b = (high - origin) / step;
    double slope, fraction;
    Py_ssize_t index;
    b = fmax(b, a + 1.0e-3);
    slope = mass / (b - a);
    a = fmax(a, 0.0);
    b = fmin(b, (double)size);
    if (b <= a || mass == 0.0) return;
    index = (Py_ssize_t)floor(a);
    fraction = a - index;
    second[index] += slope * (1.0 - fraction);
    second[index + 1] += slope * fraction;
    index = (Py_ssize_t)floor(b);
    fraction = b - index;
    second[index] -= slope * (1.0 - fraction);
    second[index + 1] -= slope * fraction;
}

PyObject *
openwd_stark_manifold_bins(PyObject *self, PyObject *args)
{
    PyObject *objects[8], *result = NULL;
    Py_buffer v[8] = {{0}};
    double conversion, fine_half, fine_step, coarse_step, missing;
    double *fine_second = NULL, *coarse_second = NULL;
    double maximum = 0.0, core, inner_total = 0.0;
    Py_ssize_t nf, nu, nl, fine_size, coarse_size, i, u, l;
    int k, has_lower, invalid = 0;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOOOOdddddOO:stark_manifold_bins",
            &objects[0], &objects[1], &objects[2], &objects[3],
            &objects[4], &objects[5], &conversion, &fine_half, &fine_step,
            &coarse_step, &missing, &objects[6], &objects[7])) return NULL;
    has_lower = objects[2] != Py_None;
    if ((objects[3] != Py_None) != has_lower) {
        PyErr_SetString(PyExc_ValueError, "lower shifts and weights must both be supplied");
        goto done;
    }
    for (k = 0; k < 8; ++k) {
        if ((k == 2 || k == 3) && !has_lower) continue;
        if (!double_array(objects[k], &v[k], k >= 6)) goto done;
        if (v[k].ndim != (k < 4 ? 2 : 1)) {
            PyErr_SetString(PyExc_ValueError, "invalid Stark array rank");
            goto done;
        }
    }
    nf = v[0].shape[0]; nu = v[0].shape[1];
    nl = has_lower ? v[2].shape[1] : 1;
    fine_size = v[6].shape[0]; coarse_size = v[7].shape[0];
    if (nf < 2 || nu < 1 || nl < 1 ||
        v[1].shape[0] != nf || v[1].shape[1] != nu ||
        (has_lower && (v[2].shape[0] != nf || v[3].shape[0] != nf || v[3].shape[1] != nl)) ||
        v[4].shape[0] != nf - 1 || v[5].shape[0] != nf - 1 ||
        fine_size < 1 || fine_size % 2 != 1 ||
        (coarse_size > 0 && coarse_size % 2 != 1) ||
        !isfinite(conversion) || !isfinite(fine_half) || fine_half < 0.0 ||
        !isfinite(fine_step) || fine_step <= 0.0 ||
        !isfinite(coarse_step) || coarse_step <= 0.0 || !isfinite(missing)) {
        PyErr_SetString(PyExc_ValueError, "inconsistent Stark grids or array shapes");
        goto done;
    }
    fine_second = PyMem_Calloc((size_t)fine_size + 3, sizeof(double));
    coarse_second = PyMem_Calloc((size_t)coarse_size + 3, sizeof(double));
    if (!fine_second || !coarse_second) { PyErr_NoMemory(); goto done; }
    core = missing;
    {
        const double *us = v[0].buf, *uw = v[1].buf;
        const double *ls = has_lower ? v[2].buf : NULL;
        const double *lw = has_lower ? v[3].buf : NULL;
        const double *p = v[4].buf, *fraction = v[5].buf;
        double *fine = v[6].buf, *coarse = v[7].buf;
        Py_BEGIN_ALLOW_THREADS
        /* Preserve the original significance cutoff relative to the largest
         * component probability, including both end-point projections. */
        for (i = 0; i < nf - 1; ++i) {
            if (!isfinite(p[i]) || !isfinite(fraction[i])) { invalid = 1; break; }
            if (p[i] <= 1.0e-12) continue;
            for (u = 0; u < nu; ++u) for (l = 0; l < nl; ++l) {
                double a = uw[i*nu+u] * (has_lower ? lw[i*nl+l] : 1.0);
                double b = uw[(i+1)*nu+u] * (has_lower ? lw[(i+1)*nl+l] : 1.0);
                double mass = 0.5*(a+b)*p[i];
                if (!isfinite(mass)) invalid = 1;
                if (mass > maximum) maximum = mass;
            }
        }
        for (i = 0; i < nf - 1 && !invalid; ++i) {
            if (p[i] <= 1.0e-12) continue;
            for (u = 0; u < nu && !invalid; ++u) for (l = 0; l < nl; ++l) {
                double a = uw[i*nu+u] * (has_lower ? lw[i*nl+l] : 1.0);
                double b = uw[(i+1)*nu+u] * (has_lower ? lw[(i+1)*nl+l] : 1.0);
                double mass = 0.5*(a+b)*p[i];
                double start, stop, low, high, ilow, ihigh, imass;
                if (mass <= 1.0e-7*maximum) continue;
                start = us[i*nu+u] - (has_lower ? ls[i*nl+l] : 0.0);
                stop = us[(i+1)*nu+u] - (has_lower ? ls[(i+1)*nl+l] : 0.0);
                stop = (start + fraction[i]*(stop-start))*conversion;
                start *= conversion;
                if (!isfinite(start) || !isfinite(stop)) { invalid = 1; break; }
                if (fabs(start) <= 0.5*fine_step && fabs(stop) <= 0.5*fine_step) {
                    core += mass;
                    continue;
                }
                low = fmin(start, stop);
                high = fmax(fmax(start, stop), low + 1.0e-6*fine_step);
                ilow = fmin(fine_half, fmax(-fine_half, low));
                ihigh = fmin(fine_half, fmax(-fine_half, high));
                imass = mass * (ihigh-ilow)/(high-low);
                inner_total += imass;
                deposit_box(ilow, ihigh, imass, fine_step, fine_size, fine_second);
                if (coarse_size) {
                    double below = mass*fmax(fmin(high, -fine_half)-low, 0.0)/(high-low);
                    double above = mass*fmax(high-fmax(low, fine_half), 0.0)/(high-low);
                    if (below > 0.0) deposit_box(low, fmin(high, -fine_half), below,
                                                coarse_step, coarse_size, coarse_second);
                    if (above > 0.0) deposit_box(fmax(low, fine_half), high, above,
                                                coarse_step, coarse_size, coarse_second);
                }
            }
        }
        if (!invalid) {
            double cumulative = 0.0;
            for (i = 0; i < fine_size; ++i) { cumulative += fine_second[i]; fine[i] = cumulative; }
            cumulative = 0.0;
            for (i = 0; i < coarse_size; ++i) { cumulative += coarse_second[i]; coarse[i] = cumulative; }
        }
        Py_END_ALLOW_THREADS
    }
    if (invalid) PyErr_SetString(PyExc_ValueError, "non-finite Stark component data");
    else result = Py_BuildValue("dd", core, inner_total);
done:
    PyMem_Free(fine_second);
    PyMem_Free(coarse_second);
    for (k = 0; k < 8; ++k) if (v[k].obj) PyBuffer_Release(&v[k]);
    return result;
}

/* Interpolate the two uniform grids directly and evaluate the impact core
 * in the same pass.  This avoids binary searches, masks, and wavelength-size
 * temporaries for each atmospheric depth.  Edge values match np.interp.
 */
static double
uniform_profile(double offset, double step, const double *mass,
                Py_ssize_t size, int clip_negative)
{
    double position = offset / step + (size - 1) / 2;
    Py_ssize_t left;
    double fraction, a, b;
    if (position <= 0.0) { left = 0; fraction = 0.0; }
    else if (position >= size - 1) { left = size - 1; fraction = 0.0; }
    else { left = (Py_ssize_t)floor(position); fraction = position - left; }
    a = mass[left];
    b = mass[left + (left < size - 1)];
    if (clip_negative) { if (a < 0.0) a = 0.0; if (b < 0.0) b = 0.0; }
    a /= step; b /= step;
    return a + fraction*(b-a);
}

PyObject *
openwd_stark_profile_finish(PyObject *self, PyObject *args)
{
    PyObject *objects[4], *result = NULL;
    Py_buffer v[4] = {{0}};
    double center, sigma, gamma, fine_step, coarse_step, fine_half;
    double core_weight, inner_mass, bound;
    const double pi = 3.1415926535897932384626433832795;
    const double ln2 = 0.69314718055994530941723212145818;
    int k;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOdddddddddO:stark_profile_finish",
            &objects[0], &objects[1], &objects[2], &center, &sigma, &gamma,
            &fine_step, &coarse_step, &fine_half, &core_weight, &inner_mass,
            &bound, &objects[3])) return NULL;
    for (k = 0; k < 4; ++k) {
        if (!double_array(objects[k], &v[k], k == 3)) goto done;
        if (v[k].ndim != 1) {
            PyErr_SetString(PyExc_ValueError, "profile arrays must be one-dimensional");
            goto done;
        }
    }
    if (v[0].shape[0] != v[3].shape[0] || v[1].shape[0] < 1 ||
        v[1].shape[0] % 2 != 1 || (v[2].shape[0] && v[2].shape[0] % 2 != 1) ||
        !isfinite(center) || !isfinite(sigma) || !isfinite(gamma) ||
        !isfinite(fine_step) || fine_step <= 0.0 ||
        !isfinite(coarse_step) || coarse_step <= 0.0 ||
        !isfinite(fine_half) || fine_half < 0.0 ||
        !isfinite(core_weight) || !isfinite(inner_mass) ||
        !isfinite(bound) || bound <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "invalid profile grids, widths, or normalization");
        goto done;
    }
    {
        double g = 2.0*sqrt(2.0*ln2)*fmax(sigma, 1.0e-12);
        double l = 2.0*fmax(gamma, 0.0);
        double width = pow(pow(g, 5.0) + 2.69269*pow(g, 4.0)*l +
            2.42843*pow(g, 3.0)*pow(l, 2.0) +
            4.47163*pow(g, 2.0)*pow(l, 3.0) +
            0.07842*g*pow(l, 4.0) + pow(l, 5.0), 0.2);
        double r = l/width;
        double mix = fmin(1.0, fmax(0.0, 1.36603*r - 0.47719*r*r + 0.11116*r*r*r));
        const double *wave = v[0].buf, *fine = v[1].buf, *coarse = v[2].buf;
        double *out = v[3].buf;
        Py_ssize_t i;
        Py_BEGIN_ALLOW_THREADS
        for (i = 0; i < v[0].shape[0]; ++i) {
            double offset = wave[i] - center;
            double z = offset/width;
            double impact = (1.0-mix)*2.0*sqrt(ln2)/(sqrt(pi)*width)*exp(-4.0*ln2*z*z)
                + mix*2.0/(pi*width)/(1.0+4.0*z*z);
            double value = core_weight*impact;
            if (isnan(offset)) { out[i] = offset; continue; }
            if (fabs(offset) <= fine_half) {
                value += uniform_profile(offset, fine_step, fine, v[1].shape[0], 1);
            } else if (v[2].shape[0]) {
                value += uniform_profile(offset, coarse_step, coarse, v[2].shape[0], 0)
                    + inner_mass*impact;
            }
            out[i] = value/bound;
        }
        Py_END_ALLOW_THREADS
    }
    Py_INCREF(Py_None); result = Py_None;
done:
    for (k = 0; k < 4; ++k) if (v[k].obj) PyBuffer_Release(&v[k]);
    return result;
}

/* np.interp on the exact coordinates step * (arange(size) - size//2).
 * A uniform-grid index avoids a binary search, but use the actual rounded
 * endpoint coordinates and interpolation operation order of the reference.
 * The fine mass is clipped before interpolation, not after it.
 */
static double
frequency_uniform_profile(double offset, double step, const double *mass,
                          Py_ssize_t size, int clip_negative)
{
    const Py_ssize_t half = size / 2;
    const double first = -half * step;
    const double last = (size - 1 - half) * step;
    Py_ssize_t left;
    double a, b, low, high, slope, value;
    if (offset <= first) {
        a = mass[0];
        if (clip_negative && a <= 0.0) a = 0.0;
        return a / step;
    }
    if (offset >= last) {
        a = mass[size - 1];
        if (clip_negative && a <= 0.0) a = 0.0;
        return a / step;
    }
    left = (Py_ssize_t)floor(offset / step + half);
    if (left < 0) left = 0;
    if (left >= size - 1) left = size - 2;
    /* Division can round across an exact grid point; recover the same
     * bracket that a search of the stored Python coordinates would find. */
    while (left > 0 && offset < (left - half) * step) --left;
    while (left < size - 2 && offset >= (left + 1 - half) * step) ++left;
    low = (left - half) * step;
    high = (left + 1 - half) * step;
    a = mass[left]; b = mass[left + 1];
    if (clip_negative) { if (a <= 0.0) a = 0.0; if (b <= 0.0) b = 0.0; }
    a /= step; b /= step;
    if (offset == low) return a;
    slope = (b - a) / (high - low);
    value = slope * (offset - low) + a;
    /* Match np.interp's treatment of non-finite endpoint values. */
    if (isnan(value)) {
        value = slope * (offset - high) + b;
        if (isnan(value) && a == b) value = a;
    }
    return value;
}

PyObject *
openwd_stark_frequency_profile_finish(PyObject *self, PyObject *args)
{
    PyObject *objects[4], *result = NULL;
    Py_buffer v[4] = {{0}};
    double center, sigma, gamma, fine_step, coarse_step, fine_half;
    double core_weight, inner_mass, bound;
    FrequencyVoigt profile;
    int k;
    (void)self;
    if (!PyArg_ParseTuple(args, "OOOdddddddddO:stark_frequency_profile_finish",
            &objects[0], &objects[1], &objects[2], &center, &sigma, &gamma,
            &fine_step, &coarse_step, &fine_half, &core_weight, &inner_mass,
            &bound, &objects[3])) return NULL;
    for (k = 0; k < 4; ++k) {
        if (!double_array(objects[k], &v[k], k == 3)) goto done;
        if (v[k].ndim != 1) {
            PyErr_SetString(PyExc_ValueError, "profile arrays must be one-dimensional");
            goto done;
        }
    }
    if (v[0].shape[0] != v[3].shape[0] || v[1].shape[0] < 1 ||
        v[1].shape[0] % 2 != 1 || (v[2].shape[0] && v[2].shape[0] % 2 != 1) ||
        !isfinite(center) || center <= 0.0 ||
        !isfinite(sigma) || !isfinite(gamma) ||
        !isfinite(fine_step) || fine_step <= 0.0 ||
        !isfinite(coarse_step) || coarse_step <= 0.0 ||
        !isfinite(fine_half) || fine_half < 0.0 ||
        !isfinite(core_weight) || !isfinite(inner_mass) ||
        !isfinite(bound) || bound <= 0.0) {
        PyErr_SetString(PyExc_ValueError, "invalid profile grids, widths, or normalization");
        goto done;
    }
    for (k = 0; k < 3; ++k) {
        if (arrays_overlap(&v[k], &v[3])) {
            PyErr_SetString(PyExc_ValueError, "profile output must not overlap inputs");
            goto done;
        }
    }
    frequency_voigt_setup(&profile, center, sigma, gamma);
    {
        const double *wave = v[0].buf, *fine = v[1].buf, *coarse = v[2].buf;
        double *out = v[3].buf;
        Py_ssize_t i;
        Py_BEGIN_ALLOW_THREADS
        for (i = 0; i < v[0].shape[0]; ++i) {
            const double offset = wave[i] - center;
            const double impact = frequency_voigt_value(&profile, wave[i]);
            double value = core_weight * impact;
            if (isnan(offset)) { out[i] = offset; continue; }
            if (fabs(offset) <= fine_half) {
                value += frequency_uniform_profile(
                    offset, fine_step, fine, v[1].shape[0], 1);
            } else if (v[2].shape[0]) {
                value += frequency_uniform_profile(
                    offset, coarse_step, coarse, v[2].shape[0], 0)
                    + inner_mass * impact;
            }
            out[i] = value / bound;
        }
        Py_END_ALLOW_THREADS
    }
    Py_INCREF(Py_None); result = Py_None;
done:
    for (k = 0; k < 4; ++k) if (v[k].obj) PyBuffer_Release(&v[k]);
    return result;
}
