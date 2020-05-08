functions{
  real y_a(t,s){
    return 0.0;
  }
}

data{
  int<lower=0> N;
  real<lower=0.0> times[N];
  int<lower=0> b_counts[N];
}
transformed_data{
  real x_r[0];
  int x_i[0];
}

parameters{
  real<lower=0.0> r1_alpha;
  real<lower=0.0> r1_beta;
}

model{
  target += poisson_lpdf(2);
}