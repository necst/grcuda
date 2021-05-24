# fromFile.p
# Make 2d surface graph and 2d heatmap graph
# parameters
# ----------
# filename The input file to use for the graph. It should be an array to triples
#          in the form of (x-coordinate, y-coordinate, z-value).
#
# outname  The output file name.
#  
# dimX The dimension of the x-coordinate.
#
# dimY The dimension of the y-coordinate.

set terminal svg size 400,200 dynamic

set title outname." Heat Plot"
set output 'graphs/'.outname.'_heat_'.dimX.'_'.dimY.'.svg'
plot filename binary record=(dimX,dimY) scan=yx with image notitle 

set title outname." Surface Plot"
set grid
set hidden3d
set output 'graphs/'.outname.'_surface_'.dimX.'_'.dimY.'.svg'
splot filename binary record=(dimX,dimY) scan=yx with lines notitle 
