#ifndef _MYRAND_H_
#define _MYRAND_H_

#define MR_MULTIPLIER 279470273 
#define MR_MODULUS 4294967291U
#define MR_DIVISOR ((double) 4294967291U)

unsigned my_rand(unsigned* seed_p) {
   long long z = *seed_p;
   z *= MR_MULTIPLIER; 
   z %= MR_MODULUS;
   *seed_p = z;
   return *seed_p;
}

double my_drand(unsigned* seed_p) {
   unsigned x = my_rand(seed_p);
   double y = x/MR_DIVISOR;
   return y;
}

#endif