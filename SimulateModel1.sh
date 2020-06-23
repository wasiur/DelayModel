NSIM=500
N=1000
B=0.1
CONVDISTNAME="Weibull"
CONVDISTPARMS="[0.50,2.75]"
DEATHDISTNAME="Weibull"
DEATHDISTPARMS="[0.75,3.75]"
DT=0.1
FINALTIME=5.0
BACKEND="GR"

julia SampleModel1.jl -n $N -b $B --nSim=$NSIM --conv-dist-name=$CONVDISTNAME \
 --conv-dist-parms=$CONVDISTPARMS --death-dist-name=$DEATHDISTNAME --death-dist-parms=$DEATHDISTPARMS \
  --dt=$DT -p --final-time=$FINALTIME --backend=$BACKEND
