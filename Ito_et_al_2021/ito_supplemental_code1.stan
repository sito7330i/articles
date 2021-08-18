// Copyright (c) 2016, Hiroki Itô
// All rights reserved.
// The source code was modified by Shun Ito to fit the model.
// -------------------------------------------------
// States (S):
// 1 survive within the survey site at juvenile
// 2 dead within the survey site at juvenile
// 3 survive outside the survey site at juvenile
// 4 dead outside the survey site at juvenile
// 5 survive within the survey site at adult
// 6 dead within the survey site at adult
// 7 survive outside the survey site at adult
// 8 dead outside the survey site at adult
// Observations (O):
// 1 discovered an alive snail at juvenile
// 2 recovered a dead marked snail at juvenile
// 3 discovered an alive snail at adult
// 4 recovered a dead marked snail at adult
// 5 undiscovered or unrecovered a marked snail
// -------------------------------------------------

functions {
  int first_capture(int[] y_i) {
    for (k in 1:size(y_i))
      if (y_i[k] != 7)
        return k;
    return 0;
  }
}

data {
  int<lower=0> nind;
  int<lower=0> n_occasions;
  int<lower=1,upper=7> y[nind, n_occasions];
  vector[nind] col_;
  int month[n_occasions];
}

transformed data {
  int n_occ_minus_1 = n_occasions - 1;
  int<lower=0,upper=n_occasions> first[nind];

  for (i in 1:nind)
    first[i] = first_capture(y[i]);
}

parameters {
  real<lower=0,upper=1> mean_sJ;
  real<lower=0,upper=1> mean_sA;
  real<lower=0,upper=1> mean_predJ;
  real<lower=0,upper=1> mean_predA;
  real<lower=0,upper=1> mean_f;
  real<lower=0,upper=1> mean_r;
  real<lower=0,upper=1> mean_p;
  real<lower=0,upper=1> mean_g;
  real beta_sJ;
  real gamma_sJ;
  real beta_sA;
  real gamma_sA;
  vector[nind] raw_epsilon_sJ;
  real<lower=0,upper=15> sigma_sJ;
  vector<lower=0,upper=1>[12] bJ;
  vector[nind] raw_epsilon_sA;
  real<lower=0,upper=15> sigma_sA;
  vector<lower=0,upper=1>[12] bA;
  vector[12] epsilon_predJ;
  real<lower=0,upper=15> sigma_predJ;
  vector[12] epsilon_predA;
  real<lower=0,upper=15> sigma_predA;
  vector[12] epsilon_g;
  real<lower=0,upper=15> sigma_g;
  real beta_p;
  real beta_r;
  vector[nind] raw_epsilon_rp;
  real<lower=0,upper=15> sigma_rp;
  vector[n_occ_minus_1] epsilon_r;
  vector[n_occ_minus_1] epsilon_p;
  real<lower=0,upper=15> sigma_r;
  real<lower=0,upper=15> sigma_p;
}

transformed parameters {
  vector<lower=0,upper=1>[n_occ_minus_1] sJ[nind];
  vector<lower=0,upper=1>[n_occ_minus_1] sA[nind];
  vector<lower=0,upper=1>[n_occ_minus_1] predJ[nind];
  vector<lower=0,upper=1>[n_occ_minus_1] predA[nind];
  vector<lower=0,upper=1>[n_occ_minus_1] F[nind];
  vector<lower=0,upper=1>[n_occ_minus_1] g[nind];
  real mu_sJ = logit(mean_sJ);
  real mu_sA = logit(mean_sA);
  vector[12] BJ = logit(bJ);
  vector[12] BA = logit(bA);
  vector[nind] epsilon_sJ;
  vector[nind] epsilon_sA;
  real mu_g = logit(mean_g);
  vector<lower=0,upper=1>[n_occ_minus_1] p_it[nind];
  vector<lower=0,upper=1>[n_occ_minus_1] r_it[nind];
  real mu_r = logit(mean_r);
  real mu_p = logit(mean_p);
  vector[nind] epsilon_rp;
  simplex[12] ps[12, nind, n_occ_minus_1];
  simplex[7] po[12, nind, n_occ_minus_1];
  
  epsilon_sJ = raw_epsilon_sJ*sqrt(sigma_sJ);
  epsilon_sA = raw_epsilon_sA*sqrt(sigma_sA);
  epsilon_rp = raw_epsilon_rp*sqrt(sigma_rp);

  
  for (i in 1:nind) {
    for (t in 1:n_occ_minus_1) {
      sJ[i,t] = inv_logit(mu_sJ + beta_sJ*col_[i]
        + 0.5*(gamma_sJ*pow(col_[i],2)) + epsilon_sJ[i] + BJ[month[t]]);
      sA[i,t] = inv_logit(mu_sA + beta_sA*col_[i]
        + 0.5*(gamma_sA*pow(col_[i],2)) + epsilon_sA[i] + BA[month[t]]);
      predJ[i,t] = mean_predJ;
      predA[i,t] = mean_predA;
      F[i,t] = mean_f;
      g[i,t] = inv_logit(mu_g + epsilon_g[month[t]]);
      r_it[i,t] = inv_logit(mu_r + beta_r*col_[i] + epsilon_rp[i] + epsilon_r[t]);
      p_it[i,t] = inv_logit(mu_p + beta_p*col_[i] + epsilon_rp[i] + epsilon_p[t]);
      
      ps[1, i, t, 1] = (1.0 - g[i,t])*sJ[i,t]*F[i,t];
      ps[1, i, t, 2] = (1.0 - g[i,t])*(1.0 - predJ[i,t])*(1.0 - sJ[i,t])*F[i,t];
      ps[1, i, t, 3] = (1.0 - g[i,t])*predJ[i,t]*(1.0 - sJ[i,t])*F[i,t];
      ps[1, i, t, 4] = g[i,t]*sJ[i,t]*F[i,t];
      ps[1, i, t, 5] = g[i,t]*(1.0 - predJ[i,t])*(1.0 - sJ[i,t])*F[i,t];
      ps[1, i, t, 6] = g[i,t]*predJ[i,t]*(1.0 - sJ[i,t])*F[i,t];
      ps[1, i, t, 7] = (1.0 - g[i,t])*sJ[i,t]*(1.0 - F[i,t]);
      ps[1, i, t, 8] = (1.0 - g[i,t])*(1.0 - predJ[i,t])*(1.0 - sJ[i,t])*(1.0 - F[i,t]);
      ps[1, i, t, 9] = (1.0 - g[i,t])*predJ[i,t]*(1.0 - sJ[i,t])*(1.0 - F[i,t]);
      ps[1, i, t, 10] = g[i,t]*sJ[i,t]*(1.0 - F[i,t]);
      ps[1, i, t, 11] = g[i,t]*(1.0 - predJ[i,t])*(1.0 - sJ[i,t])*(1.0 - F[i,t]);
      ps[1, i, t, 12] = g[i,t]*predJ[i,t]*(1.0 - sJ[i,t])*(1.0 - F[i,t]);
      ps[2, i, t, 1] = 0.0;
      ps[2, i, t, 2] = 1.0;
      ps[2, i, t, 3] = 0.0;
      ps[2, i, t, 4] = 0.0;
      ps[2, i, t, 5] = 0.0;
      ps[2, i, t, 6] = 0.0;
      ps[2, i, t, 7] = 0.0;
      ps[2, i, t, 8] = 0.0;
      ps[2, i, t, 9] = 0.0;
      ps[2, i, t, 10] = 0.0;
      ps[2, i, t, 11] = 0.0;
      ps[2, i, t, 12] = 0.0;
      ps[3, i, t, 1] = 0.0;
      ps[3, i, t, 2] = 0.0;
      ps[3, i, t, 3] = 1.0;
      ps[3, i, t, 4] = 0.0;
      ps[3, i, t, 5] = 0.0;
      ps[3, i, t, 6] = 0.0;
      ps[3, i, t, 7] = 0.0;
      ps[3, i, t, 8] = 0.0;
      ps[3, i, t, 9] = 0.0;
      ps[3, i, t, 10] = 0.0;
      ps[3, i, t, 11] = 0.0;
      ps[3, i, t, 12] = 0.0;
      ps[4, i, t, 1] = 0.0;
      ps[4, i, t, 2] = 0.0;
      ps[4, i, t, 3] = 0.0;
      ps[4, i, t, 4] = sA[i,t]*F[i,t];
      ps[4, i, t, 5] = (1.0 - predA[i,t])*(1.0 - sA[i,t])*F[i,t];
      ps[4, i, t, 6] = predA[i,t]*(1.0 - sA[i,t])*F[i,t];
      ps[4, i, t, 7] = 0.0;
      ps[4, i, t, 8] = 0.0;
      ps[4, i, t, 9] = 0.0;
      ps[4, i, t, 10] = sA[i,t]*(1.0 - F[i,t]);
      ps[4, i, t, 11] = (1.0 - predA[i,t])*(1.0 - sA[i,t])*(1.0 - F[i,t]);
      ps[4, i, t, 12] = predA[i,t]*(1.0 - sA[i,t])*(1.0 - F[i,t]);
      ps[5, i, t, 1] = 0.0;
      ps[5, i, t, 2] = 0.0;
      ps[5, i, t, 3] = 0.0;
      ps[5, i, t, 4] = 0.0;
      ps[5, i, t, 5] = 1.0;
      ps[5, i, t, 6] = 0.0;
      ps[5, i, t, 7] = 0.0;
      ps[5, i, t, 8] = 0.0;
      ps[5, i, t, 9] = 0.0;
      ps[5, i, t, 10] = 0.0;
      ps[5, i, t, 11] = 0.0;
      ps[5, i, t, 12] = 0.0;
      ps[6, i, t, 1] = 0.0;
      ps[6, i, t, 2] = 0.0;
      ps[6, i, t, 3] = 0.0;
      ps[6, i, t, 4] = 0.0;
      ps[6, i, t, 5] = 0.0;
      ps[6, i, t, 6] = 1.0;
      ps[6, i, t, 7] = 0.0;
      ps[6, i, t, 8] = 0.0;
      ps[6, i, t, 9] = 0.0;
      ps[6, i, t, 10] = 0.0;
      ps[6, i, t, 11] = 0.0;
      ps[6, i, t, 12] = 0.0;
      ps[7, i, t, 1] = 0.0;
      ps[7, i, t, 2] = 0.0;
      ps[7, i, t, 3] = 0.0;
      ps[7, i, t, 4] = 0.0;
      ps[7, i, t, 5] = 0.0;
      ps[7, i, t, 6] = 0.0;
      ps[7, i, t, 7] = (1.0 - g[i,t])*sJ[i,t];
      ps[7, i, t, 8] = (1.0 - g[i,t])*(1.0 - predJ[i,t])*(1.0 - sJ[i,t]);
      ps[7, i, t, 9] = (1.0 - g[i,t])*predJ[i,t]*(1.0 - sJ[i,t]);
      ps[7, i, t, 10] = g[i,t]*sJ[i,t];
      ps[7, i, t, 11] = g[i,t]*(1.0 - predJ[i,t])*(1.0 - sJ[i,t]);
      ps[7, i, t, 12] = g[i,t]*predJ[i,t]*(1.0 - sJ[i,t]);
      ps[8, i, t, 1] = 0.0;
      ps[8, i, t, 2] = 0.0;
      ps[8, i, t, 3] = 0.0;
      ps[8, i, t, 4] = 0.0;
      ps[8, i, t, 5] = 0.0;
      ps[8, i, t, 6] = 0.0;
      ps[8, i, t, 7] = 0.0;
      ps[8, i, t, 8] = 1.0;
      ps[8, i, t, 9] = 0.0;
      ps[8, i, t, 10] = 0.0;
      ps[8, i, t, 11] = 0.0;
      ps[8, i, t, 12] = 0.0;
      ps[9, i, t, 1] = 0.0;
      ps[9, i, t, 2] = 0.0;
      ps[9, i, t, 3] = 0.0;
      ps[9, i, t, 4] = 0.0;
      ps[9, i, t, 5] = 0.0;
      ps[9, i, t, 6] = 0.0;
      ps[9, i, t, 7] = 0.0;
      ps[9, i, t, 8] = 0.0;
      ps[9, i, t, 9] = 1.0;
      ps[9, i, t, 10] = 0.0;
      ps[9, i, t, 11] = 0.0;
      ps[9, i, t, 12] = 0.0;
      ps[10, i, t, 1] = 0.0;
      ps[10, i, t, 2] = 0.0;
      ps[10, i, t, 3] = 0.0;
      ps[10, i, t, 4] = 0.0;
      ps[10, i, t, 5] = 0.0;
      ps[10, i, t, 6] = 0.0;
      ps[10, i, t, 7] = 0.0;
      ps[10, i, t, 8] = 0.0;
      ps[10, i, t, 9] = 0.0;
      ps[10, i, t, 10] = sA[i,t];
      ps[10, i, t, 11] = (1.0 - predA[i,t])*(1.0 - sA[i,t]);
      ps[10, i, t, 12] = predA[i,t]*(1.0 - sA[i,t]);
      ps[11, i, t, 1] = 0.0;
      ps[11, i, t, 2] = 0.0;
      ps[11, i, t, 3] = 0.0;
      ps[11, i, t, 4] = 0.0;
      ps[11, i, t, 5] = 0.0;
      ps[11, i, t, 6] = 0.0;
      ps[11, i, t, 7] = 0.0;
      ps[11, i, t, 8] = 0.0;
      ps[11, i, t, 9] = 0.0;
      ps[11, i, t, 10] = 0.0;
      ps[11, i, t, 11] = 1.0;
      ps[11, i, t, 12] = 0.0;
      ps[12, i, t, 1] = 0.0;
      ps[12, i, t, 2] = 0.0;
      ps[12, i, t, 3] = 0.0;
      ps[12, i, t, 4] = 0.0;
      ps[12, i, t, 5] = 0.0;
      ps[12, i, t, 6] = 0.0;
      ps[12, i, t, 7] = 0.0;
      ps[12, i, t, 8] = 0.0;
      ps[12, i, t, 9] = 0.0;
      ps[12, i, t, 10] = 0.0;
      ps[12, i, t, 11] = 0.0;
      ps[12, i, t, 12] = 1.0;
      
      if (t<4) {
        r_it[i,t] = 0;
      }
      po[1, i, t, 1] = p_it[i,t];
      po[1, i, t, 2] = 0.0;
      po[1, i, t, 3] = 0.0;
      po[1, i, t, 4] = 0.0;
      po[1, i, t, 5] = 0.0;
      po[1, i, t, 6] = 0.0;
      po[1, i, t, 7] = 1.0 - p_it[i,t];
      po[2, i, t, 1] = 0.0;
      po[2, i, t, 2] = r_it[i,t];
      po[2, i, t, 3] = 0.0;
      po[2, i, t, 4] = 0.0;
      po[2, i, t, 5] = 0.0;
      po[2, i, t, 6] = 0.0;
      po[2, i, t, 7] = 1.0 - r_it[i,t];
      po[3, i, t, 1] = 0.0;
      po[3, i, t, 2] = 0.0;
      po[3, i, t, 3] = r_it[i,t];
      po[3, i, t, 4] = 0.0;
      po[3, i, t, 5] = 0.0;
      po[3, i, t, 6] = 0.0;
      po[3, i, t, 7] = 1.0 - r_it[i,t];
      po[4, i, t, 1] = 0.0;
      po[4, i, t, 2] = 0.0;
      po[4, i, t, 3] = 0.0;
      po[4, i, t, 4] = p_it[i,t];
      po[4, i, t, 5] = 0.0;
      po[4, i, t, 6] = 0.0;
      po[4, i, t, 7] = 1.0 - p_it[i,t];
      po[5, i, t, 1] = 0.0;
      po[5, i, t, 2] = 0.0;
      po[5, i, t, 3] = 0.0;
      po[5, i, t, 4] = 0.0;
      po[5, i, t, 5] = r_it[i,t];
      po[5, i, t, 6] = 0.0;
      po[5, i, t, 7] = 1.0 - r_it[i,t];
      po[6, i, t, 1] = 0.0;
      po[6, i, t, 2] = 0.0;
      po[6, i, t, 3] = 0.0;
      po[6, i, t, 4] = 0.0;
      po[6, i, t, 5] = 0.0;
      po[6, i, t, 6] = r_it[i,t];
      po[6, i, t, 7] = 1.0 - r_it[i,t];
      po[7, i, t, 1] = 0.0;
      po[7, i, t, 2] = 0.0;
      po[7, i, t, 3] = 0.0;
      po[7, i, t, 4] = 0.0;
      po[7, i, t, 5] = 0.0;
      po[7, i, t, 6] = 0.0;
      po[7, i, t, 7] = 1.0;
      po[8, i, t, 1] = 0.0;
      po[8, i, t, 2] = 0.0;
      po[8, i, t, 3] = 0.0;
      po[8, i, t, 4] = 0.0;
      po[8, i, t, 5] = 0.0;
      po[8, i, t, 6] = 0.0;
      po[8, i, t, 7] = 1.0;
      po[9, i, t, 1] = 0.0;
      po[9, i, t, 2] = 0.0;
      po[9, i, t, 3] = 0.0;
      po[9, i, t, 4] = 0.0;
      po[9, i, t, 5] = 0.0;
      po[9, i, t, 6] = 0.0;
      po[9, i, t, 7] = 1.0;
      po[10, i, t, 1] = 0.0;
      po[10, i, t, 2] = 0.0;
      po[10, i, t, 3] = 0.0;
      po[10, i, t, 4] = 0.0;
      po[10, i, t, 5] = 0.0;
      po[10, i, t, 6] = 0.0;
      po[10, i, t, 7] = 1.0;
      po[11, i, t, 1] = 0.0;
      po[11, i, t, 2] = 0.0;
      po[11, i, t, 3] = 0.0;
      po[11, i, t, 4] = 0.0;
      po[11, i, t, 5] = 0.0;
      po[11, i, t, 6] = 0.0;
      po[11, i, t, 7] = 1.0;
      po[12, i, t, 1] = 0.0;
      po[12, i, t, 2] = 0.0;
      po[12, i, t, 3] = 0.0;
      po[12, i, t, 4] = 0.0;
      po[12, i, t, 5] = 0.0;
      po[12, i, t, 6] = 0.0;
      po[12, i, t, 7] = 1.0;
    }
   }
}

model {
  real acc[12];
  vector[12] gam[n_occasions];
  
  mean_sJ ~ uniform(0, 1);
  mean_sA ~ uniform(0, 1);
  mean_predJ ~ uniform(0, 1);
  mean_predA ~ uniform(0, 1);
  mean_f ~ uniform(0, 1);
  mean_r ~ uniform(0, 1);
  mean_p ~ uniform(0, 1);
  mean_g ~ uniform(0, 1);
  beta_sJ ~ normal(0, 100);
  gamma_sJ ~ normal(0, 100);
  beta_sA ~ normal(0, 100);
  gamma_sA ~ normal(0, 100);
  raw_epsilon_sJ ~ normal(0, 1);
  sigma_sJ ~ uniform(0, 15);
  bJ ~ uniform(0, 1);
  raw_epsilon_sA ~ normal(0, 1);
  sigma_sA ~ uniform(0, 15);
  bA ~ uniform(0, 1);
  epsilon_predJ ~ normal(0, sqrt(sigma_predJ));
  sigma_predJ ~ uniform(0, 15);
  epsilon_predA ~ normal(0, sqrt(sigma_predA));
  sigma_predA ~ uniform(0, 15);
  epsilon_g ~ normal(0, sqrt(sigma_g));
  sigma_g ~ uniform(0, 15);
  beta_p ~ normal(0, 100);
  beta_r ~ normal(0, 100);
  raw_epsilon_rp ~ normal(0, 1);
  sigma_rp ~ uniform(0, 15);
  epsilon_r ~ normal(0, sqrt(sigma_r));
  epsilon_p ~ normal(0, sqrt(sigma_p));

  for (i in 1:nind) {
    if (first[i] > 0) {
      for (k in 1:12)
        gam[first[i], k] = (k == y[i, first[i]]);

      for (t in (first[i] + 1):n_occasions) {
        for (k in 1:12) {
          for (j in 1:12)
            acc[j] = gam[t-1, j] * ps[j, i, t-1, k]
                    * po[k, i, t-1, y[i, t]];
          gam[t, k] = sum(acc);
        }
      }
      target += log(sum(gam[n_occasions]));
    }
  }
}
