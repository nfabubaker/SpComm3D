testSDDMM: 
	g++ -ggdb -o testSDDMM miniapp/main_random_mat.cpp src/*.cpp -I/usr/include/x86_64-linux-gnu/mpich -L/usr/lib/x86_64-linux-gnu -lmpi
sddmm:
	mpic++ -ggdb -o sddmm miniapp/main.cpp src/*.cpp
	#g++ -O3 -o sddmm miniapp/main.cpp src/*.cpp -I/usr/include/x86_64-linux-gnu/mpich -L/usr/lib/x86_64-linux-gnu -lmpi
clean: 
	rm sddmm
