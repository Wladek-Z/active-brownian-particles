#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define nblock 125
#define blocksize 80000
#define baseof 2


int main(){

  long int i,j,ip,imax,sum;

  imax=1+log(blocksize)/log(baseof);

  for(j=0;j<nblock;j++){
    for(i=0;i<imax;i++){
      ip=(long int)pow(2.0,i*1.0);
      sum=j*blocksize+ip;
      printf("%ld\n",sum);
    }
  }
}