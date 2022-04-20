#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();//[1,1,1,1]

    // All zeros
    maskIsNegative = _pp_init_ones(0);//[0,0,0,0]

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float tooLarge = _pp_vset_float(9.999999f);;
  __pp_vec_int zero = _pp_vset_int(0);
  __pp_vec_int y;
  __pp_vec_int exp_ct = _pp_vset_int(0);
  __pp_mask maskAll, maskIsNegative, expZeroMask, tooLargeMask;

  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // All ones
    maskAll = _pp_init_ones();//[1,1,1,1]

    // All zeros
    maskIsNegative = _pp_init_ones(0);//[0,0,0,0]
    tooLargeMask = _pp_init_ones(0);//[0,0,0,0]

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll);
    _pp_vload_float(result, values + i, maskAll);
    _pp_vload_int(y, exponents + i, maskAll);

    // condition: exponent = 0
    expZeroMask = _pp_init_ones(0);//[0,0,0,0]
    _pp_veq_int(expZeroMask, y, zero, maskAll);
    _pp_vdiv_float(result, x, x, expZeroMask);


    int rounds = 0;
    int cntbits = 0;
    while(1)
    {
      rounds++;
      exp_ct = _pp_vset_int(rounds);
      // Set mask according to predicate
      _pp_vgt_int(maskIsNegative, y, exp_ct, maskAll);

      if (_pp_cntbits(maskIsNegative) == 0)
      {
        break;
      }

      _pp_vmult_float(result, result, x, maskIsNegative);
    }
    _pp_vgt_float(tooLargeMask, result, tooLarge, maskAll);
    _pp_vmove_float(result, tooLarge, tooLargeMask);

    if ((i+VECTOR_WIDTH) >= N)
    {
      maskAll = _pp_init_ones(N-i);
    }


    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);


  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_mask maskAll, maskIsNegative;

  float ans = 0;
  float ansArray[VECTOR_WIDTH];

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // All ones
    maskAll = _pp_init_ones();//[1,1,1,1]

    // All zeros
    maskIsNegative = _pp_init_ones(0);//[0,0,0,0]

    _pp_vload_float(x, values + i, maskAll);
    // _pp_vload_float(y, values + i + VECTOR_WIDTH, maskAll);

    int count = VECTOR_WIDTH;
    while(1)
    {
      if (count == 1)
      {
        break;
      }

      _pp_hadd_float(x, x);
      _pp_interleave_float(x, x);

      count = count/2;
    }
    // Write results back to memory
    _pp_vstore_float(ansArray, x, maskAll);
    ans += ansArray[0];

  }

  return ans;
}
















