// com_octave_accel.cc -- optional compiled kernels for the Octave-capable COM
// release files (octave/com_ieee8023_<ver>_octave_compat.m).
//
// The release files run on their own. When this file is built and on the path,
// three of their hottest loops call it instead of running interpreted, and
// return exactly what the interpreted code returns:
//
//   floating_fom  the candidate loop of FOM_rxffe_floating_taps
//   pdf_build     the ISI distribution loop of get_pdf_from_sampled_signal
//   ffe           FFE's tap loop
//
// "Exactly" is the design constraint, not an aspiration. Every operation is the
// one the interpreter performs, in the interpreter's order: matrix products go
// through xgemm with the transpose flags Octave's parser gives A'*B and A*B',
// solves through Matrix::solve exactly as the interpreter's xleftdiv calls it
// for A\b, with a fresh MatrixType, ranges through octave::range, min and max
// through Octave's own
// NaN-omitting rule, convolution either through octave::convn (what conv2
// calls) or as sparse adds in the order conv2 was measured to use, only on a
// step where every product is exact (checked on each step). Build without fused
// multiply-add (-ffp-contract=off): a fused a*b+c rounds once where the
// interpreter rounds twice. build_accel.py does both.
//
// Checked, not assumed: tests/test_octave_compat.py compares each kernel with
// the interpreted code, and a release is compared end to end with the kernels
// on and off (COM_OCTAVE_ACCEL=0 turns them off).
//
// Copyright 2026 Todd Bermensolo
// SPDX-License-Identifier: BSD-3-Clause

#include <octave/oct.h>
#include <octave/Range.h>
#include <octave/oct-convn.h>
#include <octave/lo-array-errwarn.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

static const char *const ACCEL_VERSION = "com_octave_accel 1 (2026-09-18)";

// Octave's scalar min and max, NaN omitted (octave::math::min/max, mappers.h).
static inline double
oct_min (double x, double y)
{
  return std::isnan (y) ? x : (x <= y ? x : y);
}

static inline double
oct_max (double x, double y)
{
  return std::isnan (y) ? x : (x >= y ? x : y);
}

// min(bmax(:), max(bmin(:), b)) as the interpreter evaluates it, a 1x1 bound
// broadcasting as a scalar.
static Matrix
clip (const Matrix& lo, const Matrix& hi, const Matrix& v)
{
  octave_idx_type n = v.numel ();
  Matrix out (n, 1);
  for (octave_idx_type i = 0; i < n; i++)
    {
      double l = lo.numel () == 1 ? lo(0) : lo(i);
      double h = hi.numel () == 1 ? hi(0) : hi(i);
      out(i) = oct_min (h, oct_max (l, v(i)));
    }
  return out;
}

// A\b as the interpreter computes it. libinterp's xleftdiv is not exported to
// oct-files, so this is its body: Matrix::solve with the caller's MatrixType,
// singular fallback on, and the same singular-matrix warning as the callback.
// The dimension check it starts with cannot fail here; the matrices are built
// to conform.
static void
singular_warning (double rcond)
{
  octave::warn_singular_matrix (rcond);
}

static Matrix
leftdiv (const Matrix& a, const Matrix& b, MatrixType& typ)
{
  octave_idx_type info;
  double rcond = 0.0;
  return a.solve (typ, b, info, rcond, singular_warning, true, blas_no_trans);
}

static bool
any_differ (const Matrix& a, const Matrix& b)
{
  for (octave_idx_type i = 0; i < a.numel (); i++)
    if (a(i) != b(i))
      return true;
  return false;
}

// ---------------------------------------------------------------- floating_fom
// best_FOM = com_octave_accel ('floating_fom', H, RnnS, Nb, dw, d, wmx, wmn,
//                              bmin, bmax, sigma_X2, fom_num, Nfix, cmx1, cpx,
//                              all_idx, valid_tap_locations, bank_size)
// Mirrors the inner loop of the FOM_rxffe_floating_taps patch, one candidate at
// a time. Second output true if a candidate's variance came out negative, -0 or
// NaN, where the interpreter would go complex or NaN: the caller then reruns the
// loop itself.
static octave_value_list
floating_fom (const octave_value_list& args)
{
  const Matrix H = args(1).matrix_value ();
  const Matrix RnnS = args(2).matrix_value ();
  const octave_idx_type Nb = args(3).idx_type_value ();
  const octave_idx_type dw = args(4).idx_type_value ();
  const octave_idx_type d = args(5).idx_type_value ();
  const Matrix wmx = args(6).matrix_value ();
  const Matrix wmn = args(7).matrix_value ();
  const Matrix bmin = args(8).matrix_value ();
  const Matrix bmax = args(9).matrix_value ();
  const double sigma_X2 = args(10).double_value ();
  const double fom_num = args(11).double_value ();
  const octave_idx_type Nfix = args(12).idx_type_value ();
  const double cmx1 = args(13).double_value ();
  const double cpx = args(14).double_value ();
  const Matrix all_idx = args(15).matrix_value ();
  const Matrix valid = args(16).matrix_value ();
  const double bank_size = args(17).double_value ();

  const octave_idx_type nrow = H.rows ();
  const octave_idx_type ncand = valid.numel ();
  RowVector best (ncand);
  bool fallback = false;

  // Data is moved through raw pointers throughout: Matrix::operator() checks
  // copy-on-write on every element it writes, which made copying Hs cost more
  // than Hs'*Hs itself. The arithmetic is untouched.
  const double *hp = H.data ();
  Matrix Hs;                            // one buffer, reused while n is unchanged
  std::vector<double> new_idx;
  for (octave_idx_type k = 0; k < ncand; k++)
    {
      // new_idx = sort([all_idx this:this+bank_size-1]) + cpx
      new_idx.assign (all_idx.data (), all_idx.data () + all_idx.numel ());
      const double loc = valid(k);
      for (double t = loc; t <= loc + bank_size - 1; t += 1)
        new_idx.push_back (t);
      std::sort (new_idx.begin (), new_idx.end ());
      // cols = [1:Nfix new_idx+cmx1], 1-based
      const octave_idx_type n = Nfix + static_cast<octave_idx_type> (new_idx.size ());
      std::vector<octave_idx_type> cols (n);
      for (octave_idx_type c = 0; c < Nfix; c++)
        cols[c] = c;                                           // 1:Nfix, 0-based
      for (std::size_t c = 0; c < new_idx.size (); c++)
        cols[Nfix + c] = static_cast<octave_idx_type> ((new_idx[c] + cpx) + cmx1) - 1;

      // Hs = H(:,cols); R = Hs'*Hs + RnnS(cols,cols)
      if (Hs.rows () != nrow || Hs.cols () != n)
        Hs = Matrix (nrow, n);
      double *hs = Hs.fortran_vec ();
      for (octave_idx_type c = 0; c < n; c++)
        std::copy (hp + cols[c] * nrow, hp + (cols[c] + 1) * nrow, hs + c * nrow);
      Matrix R = xgemm (Hs, Hs, blas_trans, blas_no_trans);
      for (octave_idx_type j = 0; j < n; j++)
        for (octave_idx_type i = 0; i < n; i++)
          R(i, j) = R(i, j) + RnnS(cols[i], cols[j]);
      // Hb = Hs(d+2:d+Nb+1,:); h0 = Hs(d+1,:)
      Matrix Hb (Nb, n), h0 (1, n);
      for (octave_idx_type c = 0; c < n; c++)
        {
          h0(0, c) = hs[c * nrow + d];
          for (octave_idx_type r = 0; r < Nb; r++)
            Hb(r, c) = hs[c * nrow + d + 1 + r];
        }
      // A = [R -Hb'; -Hb ib]; C = [h0 zb]; Z = A\C'
      const octave_idx_type m = n + Nb;
      Matrix A (m, m, 0.0), Ct (m, 1, 0.0);
      for (octave_idx_type j = 0; j < n; j++)
        for (octave_idx_type i = 0; i < n; i++)
          A(i, j) = R(i, j);
      for (octave_idx_type r = 0; r < Nb; r++)
        {
          for (octave_idx_type c = 0; c < n; c++)
            {
              A(c, n + r) = -Hb(r, c);
              A(n + r, c) = -Hb(r, c);
            }
          A(n + r, n + r) = 1.0;
        }
      Matrix C (1, m, 0.0);
      for (octave_idx_type c = 0; c < n; c++)
        {
          C(0, c) = h0(0, c);
          Ct(c, 0) = h0(0, c);
        }
      MatrixType typ;
      Matrix Z = leftdiv (A, Ct, typ);
      // S_inv = C*Z; wbl = [Z; 1-S_inv]/S_inv
      const double S_inv = (C * Z)(0, 0);
      Matrix wbl (m + 1, 1);
      for (octave_idx_type i = 0; i < m; i++)
        wbl(i) = Z(i) / S_inv;
      wbl(m) = (1 - S_inv) / S_inv;
      Matrix w (n, 1), b (Nb, 1);
      for (octave_idx_type i = 0; i < n; i++)
        w(i) = wbl(i);
      for (octave_idx_type i = 0; i < Nb; i++)
        b(i) = wbl(n + i);
      Matrix blim = clip (bmin, bmax, b);
      if (Nb > 0 && any_differ (b, blim))
        {
          // wl = [R, -h0'; h0, 0]\[h0'+Hb'*blim; 1]
          Matrix M2 (n + 1, n + 1, 0.0), rhs (n + 1, 1);
          for (octave_idx_type j = 0; j < n; j++)
            for (octave_idx_type i = 0; i < n; i++)
              M2(i, j) = R(i, j);
          for (octave_idx_type c = 0; c < n; c++)
            {
              M2(c, n) = -h0(0, c);
              M2(n, c) = h0(0, c);
            }
          Matrix HbTblim = xgemm (Hb, blim, blas_trans, blas_no_trans);
          for (octave_idx_type c = 0; c < n; c++)
            rhs(c) = h0(0, c) + HbTblim(c);
          rhs(n) = 1;
          MatrixType typ2;
          Matrix wl = leftdiv (M2, rhs, typ2);
          for (octave_idx_type i = 0; i < n; i++)
            w(i) = wl(i);
        }
      // wlim = min(wmx(:)*w(1+dw), max(wmn(:)*w(1+dw), w))
      const double w1 = w(dw);
      Matrix wlim (n, 1);
      for (octave_idx_type i = 0; i < n; i++)
        {
          const double hi = (wmx.numel () == 1 ? wmx(0) : wmx(i)) * w1;
          const double lo = (wmn.numel () == 1 ? wmn(0) : wmn(i)) * w1;
          wlim(i) = oct_min (hi, oct_max (lo, w(i)));
        }
      if (any_differ (w, wlim))
        {
          const double den = (h0 * wlim)(0, 0);
          for (octave_idx_type i = 0; i < n; i++)
            wlim(i) = wlim(i) / den;
          if (Nb > 0)
            {
              b = Hb * wlim;
              blim = clip (bmin, bmax, b);
            }
        }
      w = wlim;
      b = blim;
      // sigma_e = sqrt(sigma_X2*(w_tr*R*w+1+b'*b-2*w_tr*h0'-2*w_tr*Hb'*b))
      Matrix w_tr = w.transpose ();
      Matrix tw2 (1, n);
      for (octave_idx_type i = 0; i < n; i++)
        tw2(0, i) = 2 * w_tr(0, i);
      double acc = ((w_tr * R) * w)(0, 0);
      acc = acc + 1;
      acc = acc + xgemm (b, b, blas_trans, blas_no_trans)(0, 0);
      acc = acc - xgemm (tw2, h0, blas_no_trans, blas_trans)(0, 0);
      acc = acc - (xgemm (tw2, Hb, blas_no_trans, blas_trans) * b)(0, 0);
      const double var = sigma_X2 * acc;
      // negative, -0 or NaN: the interpreter's sqrt and log10 go complex or NaN
      if (std::isnan (var) || std::signbit (var))
        {
          fallback = true;
          best(k) = octave::numeric_limits<double>::NaN ();
          continue;
        }
      best(k) = 20 * std::log10 (fom_num / std::sqrt (var));
    }
  return ovl (best, fallback);
}

// True if every product of a nonzero q(j) and a nonzero y(i) is exact: each
// nonzero q(j) a power of two, and the smallest such product still normal.
static bool
exact_products (const Matrix& q, const Matrix& y)
{
  double qmin = octave::numeric_limits<double>::Inf ();
  for (octave_idx_type j = 0; j < q.numel (); j++)
    {
      const double qj = std::abs (q(j));
      if (qj == 0)
        continue;
      int e;
      if (std::frexp (qj, &e) != 0.5)
        return false;
      qmin = std::min (qmin, qj);
    }
  double ymin = octave::numeric_limits<double>::Inf ();
  for (octave_idx_type i = 0; i < y.numel (); i++)
    {
      const double yi = std::abs (y(i));
      if (yi != 0 && yi < ymin)
        ymin = yi;
    }
  return ! std::isfinite (qmin) || ! std::isfinite (ymin)
         || qmin * ymin >= std::numeric_limits<double>::min ();
}

// ------------------------------------------------------------------ pdf_build
// [y, Min] = com_octave_accel ('pdf_build', input_vector, values, prob, BinSize,
//                              y0, Min0)
// Mirrors the inlined loop of get_pdf_from_sampled_signal.
static octave_value_list
pdf_build (const octave_value_list& args)
{
  const NDArray iv = args(1).array_value ();
  const RowVector values = args(2).row_vector_value ();
  const RowVector prob = args(3).row_vector_value ();
  const double BinSize = args(4).double_value ();
  Matrix y = args(5).matrix_value ();
  double Min = args(6).double_value ();
  const octave_idx_type L = values.numel ();

  std::vector<double> rv (L);
  std::vector<octave_idx_type> bp (L);
  for (octave_idx_type k = 0; k < iv.numel (); k++)
    {
      const double a = std::abs (iv(k));
      for (octave_idx_type j = 0; j < L; j++)
        rv[j] = std::round ((a * values(j)) / BinSize);
      // q = zeros(1, numel(BinSize*rv(1):BinSize:BinSize*rv(end)))
      octave::range<double> r (BinSize * rv[0], BinSize, BinSize * rv[L - 1]);
      octave_idx_type nq = r.numel ();
      for (octave_idx_type j = 0; j < L; j++)
        {
          bp[j] = static_cast<octave_idx_type> (rv[j] - rv[0] + 1);   // 1-based
          nq = std::max (nq, bp[j]);                  // q(bp) grows q, as Octave does
        }
      Matrix q (1, nq, 0.0);
      bool distinct = true;
      for (octave_idx_type j = 1; j < L; j++)
        if (! (bp[j] - bp[j - 1] > 0))
          distinct = false;
      if (distinct)
        for (octave_idx_type j = 0; j < L; j++)
          q(bp[j] - 1) = prob(j);
      else
        {
          q(bp[0] - 1) = prob(0);
          for (octave_idx_type j = 1; j < L; j++)
            q(bp[j] - 1) = q(bp[j] - 1) + prob(j);
        }
      Min = std::round (Min + rv[0]);
      // Sparse adds reproduce conv2 only where every product q(j)*y(i) is exact,
      // because conv2 rounds some of its multiply-adds once (fused, in BLAS) and
      // some twice. A product is exact when q(j) is a power of two and the result
      // is a normal number; it is not when a distribution's far tail has sunk
      // into the subnormal range, which a few hundred 4-level steps reach
      // (0.25^511 = 2^-1022). Checked every step; otherwise convn, as conv2 does.
      if (exact_products (q, y))
        {
          const octave_idx_type ny = y.numel ();
          Matrix c (1, ny + nq - 1, 0.0);
          const double *yp = y.data ();
          const double *qp = q.data ();
          double *cp = c.fortran_vec ();
          for (octave_idx_type j = 0; j < nq; j++)       // ascending bins, as conv2
            {
              const double qj = qp[j];
              if (qj == 0)
                continue;
              for (octave_idx_type i = 0; i < ny; i++)
                cp[j + i] = qj * yp[i] + cp[j + i];
            }
          y = c;
        }
      else
        y = octave::convn (y, q, octave::convn_full);
    }
  return ovl (y, Min);
}

// ----------------------------------------------------------------------- ffe
// V0 = com_octave_accel ('ffe', C, cmx, spui, V). Mirrors FFE's tap loop: each
// nonzero tap added in place as two blocks; the scalar 0 if every tap is zero.
static octave_value_list
ffe (const octave_value_list& args)
{
  const NDArray C = args(1).array_value ();
  const double cmx = args(2).double_value ();
  const double spui = args(3).double_value ();
  const NDArray V = args(4).array_value ();
  const octave_idx_type n = V.numel ();
  ColumnVector V0;
  bool scalar = true;
  for (octave_idx_type i = 0; i < C.numel (); i++)
    {
      const double ci = C(i);
      if (ci == 0)
        continue;
      const double ishift = (static_cast<double> (i + 1) - 1 - cmx) * spui;
      // mod(ishift, n): ishift is a whole number
      long long s = static_cast<long long> (ishift) % static_cast<long long> (n);
      if (s < 0)
        s += n;
      if (scalar)
        {
          V0 = ColumnVector (n, 0.0);
          scalar = false;
        }
      double *v0 = V0.fortran_vec ();                 // raw: no copy-on-write check per element
      const double *vp = V.data ();
      for (octave_idx_type k = s; k < n; k++)
        v0[k] = vp[k - s] * ci + v0[k];
      for (octave_idx_type k = 0; k < s; k++)
        v0[k] = vp[n - s + k] * ci + v0[k];
    }
  if (scalar)
    return ovl (0.0);
  return ovl (V0);
}

DEFUN_DLD (com_octave_accel, args, ,
           "-*- texinfo -*-\n\
@deftypefn {} {@dots{} =} com_octave_accel (@var{op}, @dots{})\n\
Optional compiled kernels for the Octave-capable COM release files.\n\
@var{op} is @qcode{\"floating_fom\"}, @qcode{\"pdf_build\"}, @qcode{\"ffe\"} or\n\
@qcode{\"version\"}.  Each returns exactly what the interpreted code returns.\n\
@end deftypefn")
{
  if (args.length () < 1)
    print_usage ();
  const std::string op = args(0).string_value ();
  if (op == "floating_fom")
    return floating_fom (args);
  if (op == "pdf_build")
    return pdf_build (args);
  if (op == "ffe")
    return ffe (args);
  if (op == "version")
    return ovl (std::string (ACCEL_VERSION));
  error ("com_octave_accel: unknown op '%s'", op.c_str ());
}
