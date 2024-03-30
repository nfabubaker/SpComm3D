infile=$1
exc=$HOME/projects/3dsddmm/conv2bin
while read -r line
do
    outFN=`echo $(dirname $line)"/"$(basename $line .mtx)".bin"`
    sem -j16 $exc -o $outFN $line 
done < $infile  
sem --wait
