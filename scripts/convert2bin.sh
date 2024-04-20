infile=$1
exc=/PATH/TO/conv2bin
while read -r line
do
    outFN=`echo $(dirname $line)"/"$(basename $line .mtx)".bin"`
    $bin -o $outFN $file 
done < $infile  
