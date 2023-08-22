testSDDMM: 
	g++ -ggdb -o testSDDMM miniapp/main_random_mat.cpp src/*.cpp -I/usr/include/x86_64-linux-gnu/mpich -L/usr/lib/x86_64-linux-gnu -lmpi
runSDDMM:
	g++ -o testSDDMM miniapp/main.cpp src/*.cpp -I/usr/include/x86_64-linux-gnu/mpich
clean: 
	rm testSDDMM
