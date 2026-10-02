"""Смесь распределений для счётных данных (латентно-классовая модель), обучаемая EM-алгоритмом.

Для студента i и класса k:
    оценки (n5, n4, n3)    ~ Multinomial(n_graded_i, p_k)        — состав оценок
    неудачи y_im (m = 1..M) ~ Poisson(λ_km · e_i)                — пересданные неудачи, долги;
                                                                   e_i — число дисциплин (экспозиция)
В отличие от гауссовской смеси модель корректно описывает целые счётчики и большую массу нулей
(у 84 % студентов нет ни одной неудачи) и не вырождается на них.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp


class CountMixture:
    def __init__(self, n_components: int, n_init: int = 10, max_iter: int = 500,
                 tol: float = 1e-6, random_state: int | None = None):
        self.k = n_components
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    # ── вспомогательное ───────────────────────────────────────────────────────
    @staticmethod
    def _const(G, Y):
        """Слагаемые логарифма правдоподобия, не зависящие от параметров."""
        N = G.sum(1)
        return gammaln(N + 1) - gammaln(G + 1).sum(1) - gammaln(Y + 1).sum(1)

    def _log_joint(self, G, Y, E):
        lp = G @ np.log(self.p_).T                                   # (n, k)
        lam = E[:, None, None] * self.lam_[None]                      # (n, k, M)
        lp += (Y[:, None, :] * np.log(lam) - lam).sum(2)
        return lp + np.log(self.weights_)

    def _m_step(self, G, Y, E, R):
        nk = R.sum(0) + 1e-10
        self.weights_ = nk / nk.sum()
        self.p_ = (R.T @ G + 1e-3) / (R.T @ G + 1e-3).sum(1, keepdims=True)
        self.lam_ = (R.T @ Y + 1e-6) / (R.T @ E + 1e-6)[:, None]

    # ── обучение ──────────────────────────────────────────────────────────────
    def fit(self, G, Y, E):
        G, Y, E = (np.asarray(a, float) for a in (G, Y, E))
        rng = np.random.default_rng(self.random_state)
        const = self._const(G, Y)
        best = None
        for _ in range(self.n_init):
            R = rng.dirichlet(np.ones(self.k), size=len(G))
            self._m_step(G, Y, E, R)
            prev = -np.inf
            for it in range(self.max_iter):
                lj = self._log_joint(G, Y, E)
                ll_i = logsumexp(lj, 1)
                ll = (ll_i + const).sum()
                R = np.exp(lj - ll_i[:, None])
                self._m_step(G, Y, E, R)
                if ll - prev < self.tol * abs(ll):
                    break
                prev = ll
            if best is None or ll > best[0]:
                best = (ll, self.weights_.copy(), self.p_.copy(), self.lam_.copy(), it + 1)
        ll, self.weights_, self.p_, self.lam_, self.n_iter_ = best
        self.loglik_ = ll
        self.n_ = len(G)
        self._order(G)
        return self

    def _order(self, G):
        """Классы по возрастанию ожидаемого балла: 0 — самый слабый."""
        order = np.argsort(self.p_ @ np.array([5.0, 4.0, 3.0]))
        self.weights_, self.p_, self.lam_ = self.weights_[order], self.p_[order], self.lam_[order]

    # ── предсказание и критерии ───────────────────────────────────────────────
    def predict_proba(self, G, Y, E):
        G, Y, E = (np.asarray(a, float) for a in (G, Y, E))
        lj = self._log_joint(G, Y, E)
        return np.exp(lj - logsumexp(lj, 1)[:, None])

    def predict(self, G, Y, E):
        return self.predict_proba(G, Y, E).argmax(1)

    @property
    def n_params(self):
        return (self.k - 1) + self.k * (self.p_.shape[1] - 1) + self.k * self.lam_.shape[1]

    def bic(self):
        return -2 * self.loglik_ + self.n_params * np.log(self.n_)

    def icl(self, G, Y, E):
        """BIC + штраф за неуверенное отнесение (энтропия апостериорных вероятностей)."""
        R = self.predict_proba(G, Y, E)
        return self.bic() - 2 * (R * np.log(R + 1e-300)).sum()
