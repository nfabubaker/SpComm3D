## input file: a text file n lines, each line has the full path of a sparse matrix in MM format.
infile=$1
##/path/to/conv2bin
exc=../build/conv2bin
while read -r line
do
    outFN=`echo $(dirname $line)"/"$(basename $line .mtx)".bin"`
    sem -j16 $exc -o $outFN $line 
done < $infile  
sem --wait
