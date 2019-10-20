// The source code was modified by Shun Ito to fit our model.
// Copyright (c) 2016, Hiroki Itô
// All rights reserved.
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
// 2 recovered a dead  snail at juvenile
// 3 discovered an alive snail at adult
// 4 recovered a dead  snail at adult
// 5 cannot find a snail
// -------------------------------------------------

functions {
  int first_capture(int[] y_i) {
    for (k in 1:size(y_i))
      if (y_i[k] != 5)
        return k;
    return 0;
  }
  
  real check_positive(real equ) {
    if (equ >= 0) return equ;
    else return 0;
  }
}

data {
  int<lower=0> nind;  
  int<lower=0> n_occasions; 
  int<lower=0> t_month; 
  int<lower=1,upper=5> y[nind, n_occasions]; //encounter histry
  vector[nind] col_; 
  int monthes[t_month]; 
  int<lower=1> num_knots;
  vector[num_knots] knots;  
  int<lower=1> spline_degree;
  int<lower=0> n_sim;
  vector[n_sim] col_sim;
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
  real<lower=0,upper=1> mean_f; 
  vector<lower=0,upper=1>[n_occ_minus_1] r; 
  vector<lower=0,upper=1>[n_occ_minus_1] p; 
  vector<lower=0,upper=1>[12] g; 
  vector[nind] raw_epsilon_J;
  real<lower=0,upper=15> sigma_J; 
  vector<lower=0,upper=1>[12] bJ; 
  vector[nind] raw_epsilon_A; 
  real<lower=0,upper=15> sigma_A; 
  vector<lower=0,upper=1>[12] bA; 
  vector[spline_degree] raw_beta_j;
  vector[spline_degree] raw_beta_a;
  vector[num_knots] bp_J;
  vector[num_knots] bp_A;
  real<lower=0, upper=15> sigmab_J;
  real<lower=0, upper=15> sigmab_A;
  real beta_p;
  real beta_r;
  vector[nind] raw_epsilon_rp;
  real<lower=0,upper=15> sigma_rp;
}

transformed parameters {
  vector<lower=0,upper=1>[n_occ_minus_1] sJ[nind];
  vector<lower=0,upper=1>[n_occ_minus_1] sA[nind];
  vector<lower=0,upper=1>[n_occ_minus_1] F[nind];
  vector<lower=0,upper=1>[n_occ_minus_1] growth[nind];
  vector<lower=0,upper=1>[t_month] every_sJ[nind];
  vector<lower=0,upper=1>[t_month] every_sA[nind];
  vector<lower=0,upper=1>[t_month] every_F[nind];
  vector<lower=0,upper=1>[t_month] every_g[nind];
  vector[12] BJ = logit(bJ); 
  vector[12] BA = logit(bA); 
  vector[nind] epsilon_J;
  vector[nind] epsilon_A;
  vector<lower=0,upper=1>[n_occ_minus_1] p_it[nind];
  vector<lower=0,upper=1>[n_occ_minus_1] r_it[nind];
  vector[n_occ_minus_1] mu_r = logit(r);
  vector[n_occ_minus_1] mu_p = logit(p);
  vector[nind] epsilon_rp;
  simplex[8] ps[8, nind, n_occ_minus_1];
  simplex[5] po[8, nind, n_occ_minus_1];
  vector[spline_degree+1] beta_J;
  vector[spline_degree+1] beta_A;
  vector[nind] fj;
  vector[nind] fa;
  vector[num_knots] z[nind];
  vector[spline_degree+1] x[nind];
  epsilon_J = raw_epsilon_J*sqrt(sigma_J);
  epsilon_A = raw_epsilon_A*sqrt(sigma_A);
  epsilon_rp = raw_epsilon_rp*sqrt(sigma_rp);
  
  for (t in 1:spline_degree+1){
    if(t==1) {
      beta_J[t] = logit(mean_sJ);
      beta_A[t] = logit(mean_sA);
    } else {
      beta_J[t] = raw_beta_j[t-1];
      beta_A[t] = raw_beta_a[t-1];
    }
  }
    
  for (i in 1:nind){
    for (k in 1:num_knots) z[i,k] = pow(check_positive(col_[i]-knots[k]), spline_degree);
    for (l in 1:spline_degree+1) x[i,l] = pow(col_[i], l-1);
    fj[i] = dot_product(beta_J, x[i,]) + dot_product(bp_J, z[i,]);
    fa[i] = dot_product(beta_A, x[i,]) + dot_product(bp_A, z[i,]);
  }
  
  for (i in 1:nind) {
    for (t in 1:t_month) {
      every_sJ[i,t] = inv_logit(fj[i] + epsilon_J[i] + BJ[monthes[t]]);
      every_sA[i,t] = inv_logit(fa[i] + epsilon_A[i] + BA[monthes[t]]);
      every_F[i,t] = mean_f;
      every_g[i,t] = g[monthes[t]];
    }
    for (t in 1:n_occ_minus_1) {
      r_it[i,t] = inv_logit(mu_r[t] + beta_r*col_[i] + epsilon_rp[i]);
      p_it[i,t] = inv_logit(mu_p[t] + beta_p*col_[i] + epsilon_rp[i]);
      if (t<17) {
        sJ[i,t] = every_sJ[i,t];
        sA[i,t] = every_sA[i,t];
        F[i,t] = every_F[i,t];
        growth[i,t] = every_g[i,t];
      } else if(t==17){
        sJ[i,t] = every_sJ[i,17]*every_sJ[i,18];
        sA[i,t] = every_sA[i,17]*every_sA[i,18];
        F[i,t] = every_F[i,17]*every_F[i,18];
        growth[i,t] = every_g[i,17]*every_g[i,18];
      } else if(t>17){
        sJ[i,t] = every_sJ[i,t+1];
        sA[i,t] = every_sA[i,t+1];
        F[i,t] = every_F[i,t+1];
        growth[i,t] = every_g[i,t+1];
      }
    }
  }
  
  for (i in 1:nind) {
    for (t in 1:n_occ_minus_1) {
      ps[1, i, t, 1] = (1.0 - growth[i,t]) * F[i,t] * sJ[i,t];
      ps[1, i, t, 2] = (1.0 - growth[i,t]) * (1.0 - F[i,t]) * sJ[i,t];
      ps[1, i, t, 3] = growth[i,t] * F[i,t] * sJ[i,t];
      ps[1, i, t, 4] = growth[i,t] * (1.0 - F[i,t]) * sJ[i,t];
      ps[1, i, t, 5] = (1.0 - growth[i,t]) * F[i,t] * (1.0 - sJ[i,t]);
      ps[1, i, t, 6] = (growth[i,t]) * F[i,t] * (1.0 - sJ[i,t]);
      ps[1, i, t, 7] = (1.0 - growth[i,t]) * (1.0 - F[i,t]) * (1.0 - sJ[i,t]);
      ps[1, i, t, 8] = (growth[i,t]) * (1.0 - F[i,t]) * (1.0 - sJ[i,t]);
      ps[2, i, t, 1] = 0.0;
      ps[2, i, t, 2] = (1.0 - growth[i,t]) * sJ[i,t];
      ps[2, i, t, 3] = 0.0;
      ps[2, i, t, 4] = growth[i,t] * sJ[i,t];
      ps[2, i, t, 5] = 0.0;
      ps[2, i, t, 6] = 0.0;
      ps[2, i, t, 7] = (1.0 - growth[i,t]) * (1.0 - sJ[i,t]);
      ps[2, i, t, 8] = growth[i,t] * (1.0 - sJ[i,t]);
      ps[3, i, t, 1] = 0.0;
      ps[3, i, t, 2] = 0.0;
      ps[3, i, t, 3] = F[i,t] * sA[i,t];
      ps[3, i, t, 4] = (1.0 - F[i,t]) * sA[i,t];
      ps[3, i, t, 5] = 0.0;
      ps[3, i, t, 6] = F[i,t] * (1.0 - sA[i,t]);
      ps[3, i, t, 7] = 0.0;
      ps[3, i, t, 8] = (1.0 - F[i,t]) * (1.0 - sA[i,t]);
      ps[4, i, t, 1] = 0.0;
      ps[4, i, t, 2] = 0.0;
      ps[4, i, t, 3] = 0.0;
      ps[4, i, t, 4] = sA[i,t];
      ps[4, i, t, 5] = 0.0;
      ps[4, i, t, 6] = 0.0;
      ps[4, i, t, 7] = 0.0;
      ps[4, i, t, 8] = (1.0 - sA[i,t]);
      ps[5, i, t, 1] = 0.0;
      ps[5, i, t, 2] = 0.0;
      ps[5, i, t, 3] = 0.0;
      ps[5, i, t, 4] = 0.0;
      ps[5, i, t, 5] = 1.0;
      ps[5, i, t, 6] = 0.0;
      ps[5, i, t, 7] = 0.0;
      ps[5, i, t, 8] = 0.0;
      ps[6, i, t, 1] = 0.0;
      ps[6, i, t, 2] = 0.0;
      ps[6, i, t, 3] = 0.0;
      ps[6, i, t, 4] = 0.0;
      ps[6, i, t, 5] = 0.0;
      ps[6, i, t, 6] = 1.0;
      ps[6, i, t, 7] = 0.0;
      ps[6, i, t, 8] = 0.0;
      ps[7, i, t, 1] = 0.0;
      ps[7, i, t, 2] = 0.0;
      ps[7, i, t, 3] = 0.0;
      ps[7, i, t, 4] = 0.0;
      ps[7, i, t, 5] = 0.0;
      ps[7, i, t, 6] = 0.0;
      ps[7, i, t, 7] = 1.0;
      ps[7, i, t, 8] = 0.0;
      ps[8, i, t, 1] = 0.0;
      ps[8, i, t, 2] = 0.0;
      ps[8, i, t, 3] = 0.0;
      ps[8, i, t, 4] = 0.0;
      ps[8, i, t, 5] = 0.0;
      ps[8, i, t, 6] = 0.0;
      ps[8, i, t, 7] = 0.0;
      ps[8, i, t, 8] = 1.0;
      
      po[1, i, t, 1] = p[t];
      po[1, i, t, 2] = 0.0;
      po[1, i, t, 3] = 0.0;
      po[1, i, t, 4] = 0.0;
      po[1, i, t, 5] = 1.0 - p[t];
      po[2, i, t, 1] = 0.0;
      po[2, i, t, 2] = 0.0;
      po[2, i, t, 3] = 0.0;
      po[2, i, t, 4] = 0.0;
      po[2, i, t, 5] = 1.0;
      po[3, i, t, 1] = 0.0;
      po[3, i, t, 2] = 0.0;
      po[3, i, t, 3] = p[t];
      po[3, i, t, 4] = 0.0;
      po[3, i, t, 5] = 1.0 - p[t];
      po[4, i, t, 1] = 0.0;
      po[4, i, t, 2] = 0.0;
      po[4, i, t, 3] = 0.0;
      po[4, i, t, 4] = 0.0;
      po[4, i, t, 5] = 1.0;
      po[5, i, t, 1] = 0.0;
      po[5, i, t, 2] = r[t];
      po[5, i, t, 3] = 0.0;
      po[5, i, t, 4] = 0.0;
      po[5, i, t, 5] = 1.0 - r[t];
      po[6, i, t, 1] = 0.0;
      po[6, i, t, 2] = 0.0;
      po[6, i, t, 3] = 0.0;
      po[6, i, t, 4] = r[t];
      po[6, i, t, 5] = 1.0 - r[t];
      po[7, i, t, 1] = 0.0;
      po[7, i, t, 2] = 0.0;
      po[7, i, t, 3] = 0.0;
      po[7, i, t, 4] = 0.0;
      po[7, i, t, 5] = 1.0;
      po[8, i, t, 1] = 0.0;
      po[8, i, t, 2] = 0.0;
      po[8, i, t, 3] = 0.0;
      po[8, i, t, 4] = 0.0;
      po[8, i, t, 5] = 1.0;
      }
   }
}

model {
  real acc[8];
  vector[8] gam[n_occasions];
  
  mean_sJ ~ uniform(0, 1);
  mean_sA ~ uniform(0, 1);
  mean_f ~ uniform(0, 1);
  r ~ uniform(0, 1);
  p ~ uniform(0, 1);
  g ~ uniform(0, 1);
  raw_epsilon_J ~ normal(0, 1);
  sigma_J ~ uniform(0, 15);
  bJ ~ uniform(0, 1);
  raw_epsilon_A ~ normal(0, 1);
  sigma_A ~ uniform(0, 15);
  bA ~ uniform(0, 1);
  raw_beta_j ~ normal(0, 100);
  raw_beta_a ~ normal(0, 100);
  bp_J ~ normal(0, sqrt(sigmab_J));
  sigmab_J ~ inv_gamma(0.001, 0.001);
  bp_A ~ normal(0, sqrt(sigmab_A));
  sigmab_A ~ inv_gamma(0.001, 0.001);
  beta_p ~ normal(0, 100);
  beta_r ~ normal(0, 100);
  raw_epsilon_rp ~ normal(0, 1);
  sigma_rp ~ uniform(0, 15);

  for (i in 1:nind) {
    if (first[i] > 0) {
      for (k in 1:8)
        gam[first[i], k] = (k == y[i, first[i]]);

      for (t in (first[i] + 1):n_occasions) {
        for (k in 1:8) {
          for (j in 1:8)
            acc[j] = gam[t-1, j] * ps[j, i, t-1, k]
                    * po[k, i, t-1, y[i, t]];
          gam[t, k] = sum(acc);
        }
      }
      target += log(sum(gam[n_occasions]));
    }
  }
}

generated quantities {
  vector[n_sim] phi_sJ;
  vector[n_sim] phi_sA;
  vector[num_knots] z_sim[n_sim];
  vector[spline_degree+1] x_sim[n_sim];
  
  for (i in 1:n_sim){
    for (k in 1:num_knots) z_sim[i,k] = pow(check_positive(col_sim[i]-knots[k]), spline_degree);
    for (l in 1:spline_degree+1) x_sim[i,l] = pow(col_sim[i], l-1);
    phi_sJ[i] = inv_logit(dot_product(beta_J, x_sim[i,]) + dot_product(bp_J, z_sim[i,]));
    phi_sA[i] = inv_logit(dot_product(beta_A, x_sim[i,]) + dot_product(bp_A, z_sim[i,]));
  }
}
