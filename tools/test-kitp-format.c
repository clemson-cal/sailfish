#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int countlines_bin(char* filename)
{
    FILE* pFile = fopen(filename, "rb");
    fseek(pFile, 0, SEEK_END); // seek to end of file
    size_t size = ftell(pFile); // get current file pointer
    fseek(pFile, 0, SEEK_SET);
    fclose(pFile);
    return (size / 5 / sizeof(double));
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        printf("Please input file name\n");
        return (0);
    }

    char filename[256];
    if (argv[1]) {
        strcpy(filename, argv[1]);
    }
    int Npts = countlines_bin(filename);
    int Nzones = 20;

    printf("\nFile = %s has %d zones.\n", filename, Npts);
    printf("\nFirst %d zones:\n\n", Nzones);
    printf("x\t\ty\t\tSigma\t\tvx\t\tvy\n");

    FILE* pFile = fopen(filename, "r");
    int i;

    for (i = 0; i < Nzones; ++i) {
        double x, y, S, vx, vy;
        fread(&x, sizeof(double), 1, pFile);
        fread(&y, sizeof(double), 1, pFile);
        fread(&S, sizeof(double), 1, pFile);
        fread(&vx, sizeof(double), 1, pFile);
        fread(&vy, sizeof(double), 1, pFile);
        printf("%.2e\t%.2e\t%.2e\t%.2e\t%.2e\n", x, y, S, vx, vy);
    }

    fclose(pFile);
}
