# makeGraphs.sh
# Use GNUPlot to make graphs for existing data files from Jacobi iteration.
# Parameter $1 The side length of the square PDE discretization square grid.
if [ "$1" == "" ]
then
    dimX=20
    dimY=20
else
    dimX=$1
    dimY=$1
fi

echo "Graphing data/values.dat" 
gnuplot -e "filename = 'data/values.dat'" -e "outname = 'Values'" \
        -e "dimX = '$dimX'" -e "dimY = '$dimY'" fromFile.p
echo "Graphing data/errors.dat"
gnuplot -e "filename = 'data/errors.dat'" -e "outname = 'Errors'" \
        -e "dimX = '$dimX'" -e "dimY = '$dimY'" fromFile.p
echo "Graphing data/log10RelErrors.dat"
gnuplot -e "filename = 'data/log10RelErrors.dat'" -e "outname = 'Log10RelErrors'" \
        -e "dimX = '$dimX'" -e "dimY = '$dimY'" fromFile.p
