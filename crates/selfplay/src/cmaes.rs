//! A small, self-contained (μ/μ_w, λ)-CMA-ES with full covariance (Hansen, "The CMA Evolution Strategy:
//! A Tutorial", 2016, default strategy parameters). Minimises. The whole state (mean, step size,
//! covariance, evolution paths, RNG) serialises with serde, so a run resumes bit for bit.
//!
//! Bounds are the caller's business: the tuner searches the unit cube and [`reflect`]s samples into it
//! before evaluation, then passes the reflected points to [`Cma::tell`] (Lamarckian repair).

use serde::{Deserialize, Serialize};
use tetris_sim::SplitMix;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cma {
    pub n: usize,
    pub lambda: usize,
    pub mean: Vec<f64>,
    pub sigma: f64,
    /// Covariance, row-major `n × n`.
    pub cov: Vec<f64>,
    pub pc: Vec<f64>,
    pub ps: Vec<f64>,
    /// Completed generations (`tell` calls).
    pub generation: u64,
    pub rng: u64,
    /// Eigenvectors (columns, row-major `n × n`) and sqrt-eigenvalues of `cov`.
    #[serde(skip)]
    basis: Vec<f64>,
    #[serde(skip)]
    scale: Vec<f64>,
}

/// Strategy parameters, derived from `n` and `λ`.
struct Params {
    mu: usize,
    w: Vec<f64>,
    mueff: f64,
    cc: f64,
    cs: f64,
    c1: f64,
    cmu: f64,
    damps: f64,
    chi_n: f64,
}

impl Params {
    fn new(n: usize, lambda: usize) -> Params {
        let nf = n as f64;
        let mu = lambda / 2;
        let raw: Vec<f64> = (1..=mu).map(|i| (mu as f64 + 0.5).ln() - (i as f64).ln()).collect();
        let sum: f64 = raw.iter().sum();
        let w: Vec<f64> = raw.iter().map(|x| x / sum).collect();
        let mueff = 1.0 / w.iter().map(|x| x * x).sum::<f64>();
        let cc = (4.0 + mueff / nf) / (nf + 4.0 + 2.0 * mueff / nf);
        let cs = (mueff + 2.0) / (nf + mueff + 5.0);
        let c1 = 2.0 / ((nf + 1.3).powi(2) + mueff);
        let cmu = (1.0 - c1).min(2.0 * (mueff - 2.0 + 1.0 / mueff) / ((nf + 2.0).powi(2) + mueff));
        let damps = 1.0 + 2.0 * (((mueff - 1.0) / (nf + 1.0)).sqrt() - 1.0).max(0.0) + cs;
        let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));
        Params { mu, w, mueff, cc, cs, c1, cmu, damps, chi_n }
    }
}

/// The default population size `4 + ⌊3 ln n⌋`.
pub fn default_lambda(n: usize) -> usize {
    4 + (3.0 * (n as f64).ln()).floor() as usize
}

/// Fold `x` into `[0, 1]` by reflection at the bounds (periodic, so any real maps inside).
pub fn reflect(x: f64) -> f64 {
    let y = x.rem_euclid(2.0);
    if y > 1.0 { 2.0 - y } else { y }
}

impl Cma {
    pub fn new(mean: Vec<f64>, sigma: f64, lambda: usize, seed: u64) -> Cma {
        let n = mean.len();
        assert!(n >= 1 && lambda >= 4, "need n ≥ 1 and λ ≥ 4");
        let mut cov = vec![0.0; n * n];
        for i in 0..n {
            cov[i * n + i] = 1.0;
        }
        let mut c = Cma {
            n,
            lambda,
            mean,
            sigma,
            cov,
            pc: vec![0.0; n],
            ps: vec![0.0; n],
            generation: 0,
            rng: seed,
            basis: Vec::new(),
            scale: Vec::new(),
        };
        c.decompose();
        c
    }

    /// Rebuild the eigendecomposition (call after deserialising).
    pub fn decompose(&mut self) {
        let (vals, vecs) = jacobi_eigen(&self.cov, self.n);
        self.scale = vals.iter().map(|v| v.max(1e-20).sqrt()).collect();
        self.basis = vecs;
    }

    fn gauss(&mut self) -> f64 {
        let mut r = SplitMix(self.rng);
        let u1 = ((r.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let u2 = r.unit_f64();
        self.rng = r.0;
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// λ new candidates `m + σ · B · D · z`.
    pub fn ask(&mut self) -> Vec<Vec<f64>> {
        if self.basis.is_empty() {
            self.decompose();
        }
        let n = self.n;
        (0..self.lambda)
            .map(|_| {
                let z: Vec<f64> = (0..n).map(|_| self.gauss()).collect();
                (0..n)
                    .map(|i| {
                        let y: f64 = (0..n).map(|j| self.basis[i * n + j] * self.scale[j] * z[j]).sum();
                        self.mean[i] + self.sigma * y
                    })
                    .collect()
            })
            .collect()
    }

    /// Update from evaluated candidates (`fitness` lower is better; `xs` may be repaired samples).
    pub fn tell(&mut self, xs: &[Vec<f64>], fitness: &[f64]) {
        assert_eq!(xs.len(), fitness.len());
        assert!(xs.len() >= 2 * Params::new(self.n, self.lambda).mu, "too few candidates");
        let n = self.n;
        let p = Params::new(n, self.lambda);
        let mut order: Vec<usize> = (0..xs.len()).collect();
        // Stable and total: ties keep candidate order.
        order.sort_by(|&a, &b| fitness[a].total_cmp(&fitness[b]));
        let old = self.mean.clone();
        let ys: Vec<Vec<f64>> =
            order[..p.mu].iter().map(|&k| (0..n).map(|i| (xs[k][i] - old[i]) / self.sigma).collect()).collect();
        let yw: Vec<f64> = (0..n).map(|i| (0..p.mu).map(|k| p.w[k] * ys[k][i]).sum()).collect();
        for i in 0..n {
            self.mean[i] = old[i] + self.sigma * yw[i];
        }
        // C^{-1/2} · y_w = B · D^{-1} · Bᵀ · y_w
        let bt: Vec<f64> = (0..n).map(|j| (0..n).map(|i| self.basis[i * n + j] * yw[i]).sum::<f64>()).collect();
        let inv: Vec<f64> =
            (0..n).map(|i| (0..n).map(|j| self.basis[i * n + j] * bt[j] / self.scale[j]).sum()).collect();
        let k_s = (p.cs * (2.0 - p.cs) * p.mueff).sqrt();
        for (ps, x) in self.ps.iter_mut().zip(&inv) {
            *ps = (1.0 - p.cs) * *ps + k_s * x;
        }
        let ps_norm = self.ps.iter().map(|x| x * x).sum::<f64>().sqrt();
        let g = (self.generation + 1) as f64;
        let hsig = ps_norm / (1.0 - (1.0 - p.cs).powf(2.0 * g)).sqrt() / p.chi_n < 1.4 + 2.0 / (n as f64 + 1.0);
        let k_c = (p.cc * (2.0 - p.cc) * p.mueff).sqrt();
        for (pc, y) in self.pc.iter_mut().zip(&yw) {
            *pc = (1.0 - p.cc) * *pc + if hsig { k_c * y } else { 0.0 };
        }
        let dh = if hsig { 0.0 } else { p.cc * (2.0 - p.cc) };
        for i in 0..n {
            for j in 0..=i {
                let rank_mu: f64 = (0..p.mu).map(|k| p.w[k] * ys[k][i] * ys[k][j]).sum();
                let v = (1.0 - p.c1 - p.cmu) * self.cov[i * n + j]
                    + p.c1 * (self.pc[i] * self.pc[j] + dh * self.cov[i * n + j])
                    + p.cmu * rank_mu;
                self.cov[i * n + j] = v;
                self.cov[j * n + i] = v;
            }
        }
        self.sigma *= ((p.cs / p.damps) * (ps_norm / p.chi_n - 1.0)).exp();
        self.generation += 1;
        self.decompose();
    }

    /// Largest and smallest standard deviation along the coordinate axes (`σ · sqrt(C_ii)`).
    pub fn axis_sd(&self) -> (f64, f64) {
        let sd = (0..self.n).map(|i| self.sigma * self.cov[i * self.n + i].sqrt());
        sd.fold((0.0f64, f64::INFINITY), |(hi, lo), s| (hi.max(s), lo.min(s)))
    }
}

/// Eigenvalues and eigenvectors (columns of a row-major `n × n`) of a symmetric matrix, cyclic Jacobi.
pub fn jacobi_eigen(m: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = m.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i * n + j].powi(2))
            .sum();
        let diag: f64 = (0..n).map(|i| a[i * n + i].powi(2)).sum();
        if off <= 1e-30 * diag.max(1e-300) {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i * n + i]).collect(), v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_reconstructs() {
        let n = 5;
        let mut r = SplitMix(3);
        let g: Vec<f64> = (0..n * n).map(|_| r.unit() as f64 - 0.5).collect();
        // m = g gᵀ + I (symmetric positive definite)
        let m: Vec<f64> = (0..n * n)
            .map(|k| {
                let (i, j) = (k / n, k % n);
                (0..n).map(|l| g[i * n + l] * g[j * n + l]).sum::<f64>() + if i == j { 1.0 } else { 0.0 }
            })
            .collect();
        let (vals, v) = jacobi_eigen(&m, n);
        for i in 0..n {
            for j in 0..n {
                let x: f64 = (0..n).map(|l| v[i * n + l] * vals[l] * v[j * n + l]).sum();
                assert!((x - m[i * n + j]).abs() < 1e-9, "({i},{j}) {x} vs {}", m[i * n + j]);
            }
        }
    }

    fn minimise(f: impl Fn(&[f64]) -> f64, x0: Vec<f64>, sigma: f64, gens: usize) -> (f64, Cma) {
        let n = x0.len();
        let mut c = Cma::new(x0, sigma, default_lambda(n), 42);
        let mut best = f64::INFINITY;
        for _ in 0..gens {
            let xs = c.ask();
            let fs: Vec<f64> = xs.iter().map(|x| f(x)).collect();
            best = fs.iter().copied().fold(best, f64::min);
            c.tell(&xs, &fs);
        }
        (best, c)
    }

    #[test]
    fn solves_sphere_and_rosenbrock() {
        let sphere = |x: &[f64]| x.iter().map(|v| (v - 0.3).powi(2)).sum::<f64>();
        let (b, _) = minimise(sphere, vec![1.0; 10], 0.5, 300);
        assert!(b < 1e-10, "sphere {b}");
        let rosen = |x: &[f64]| x.windows(2).map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2)).sum();
        let (b, _) = minimise(rosen, vec![0.0; 6], 0.5, 1500);
        assert!(b < 1e-8, "rosenbrock {b}");
    }

    #[test]
    fn state_roundtrip_resumes_identically() {
        let f = |x: &[f64]| x.iter().enumerate().map(|(i, v)| (i + 1) as f64 * v * v).sum::<f64>();
        let mut a = Cma::new(vec![0.5; 8], 0.3, 12, 9);
        for _ in 0..5 {
            let xs = a.ask();
            let fs: Vec<f64> = xs.iter().map(|x| f(x)).collect();
            a.tell(&xs, &fs);
        }
        let mut b: Cma = serde_json::from_str(&serde_json::to_string(&a).unwrap()).unwrap();
        b.decompose();
        assert_eq!(a.ask(), b.ask());
    }

    #[test]
    fn reflect_folds_into_unit() {
        for (x, want) in [(0.25, 0.25), (1.25, 0.75), (-0.25, 0.25), (2.5, 0.5), (-1.75, 0.25)] {
            assert!((reflect(x) - want).abs() < 1e-12, "{x}");
        }
    }
}
