
PROGRAM=cholesky

export IFS=";"

THREADS="01;02;03;04;05;06;07;08;09;10;11;12"
MSIZES="1000"
BSIZES="10"

for MS in $MSIZES; do
  for BS in $BSIZES; do
    for thread in $THREADS; do
      ./$PROGRAM $MS $BS $thread
    done
  done
done
