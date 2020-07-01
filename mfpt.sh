FINALTIME=1.0
N=2000
CONVDISTNAME="InverseGamma"
CONVDISTPARMS="1.75,4.25"
DEATHDISTNAME="Weibull"
DEATHDISTPARMS="1.5,3.75"
ENDTIME=50.0

julia MeanFirstPassageTime.jl --final-time=$FINALTIME -n=$N --conv-dist-name=$CONVDISTNAME \
 --conv-dist-parms=$CONVDISTPARMS --death-dist-name=$DEATHDISTNAME --death-dist-parms=$DEATHDISTPARMS \
 --end-time=$ENDTIME
