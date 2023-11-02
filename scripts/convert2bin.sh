infile=$1
exc=/home/nabil/projects/3dsddmm/conv2bin
while read -r line
do
    outFN=`echo $(dirname $line)"/"$(basename $line .mtx)".bin"`
    $bin -o $outFN $file 
done < $infile  
