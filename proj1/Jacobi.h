/* Simple multigrid solver on unit square, using Jacobi smoother
 * (c) 2014 Philipp Neumann, TUM I-5
 */

#ifndef _JACOBI_H_
#define _JACOBI_H_

#include "Definitions.h"
#include <stdio.h>

void printField(const FLOAT *field)
{
    int nx = 3;
    for(int i = 0; i < nx + 2; i++) {
        for(int j = 0; j < nx + 2; j++){
            printf("%.3f ", field[i*(nx+2) + j]);
        }
        printf("\n");
    }
    printf("\n");
}


// carries out a jacobi step, reading from val and writing to _writeGrid.
class Jacobi
{
public:
  Jacobi(const unsigned int nx, const unsigned int ny)
      : _nx(nx), _ny(ny), _X(getX(nx, ny)), _Y(getY(nx, ny)), _RHS(getRHS(nx, ny)) {}
  ~Jacobi() {}

  void iterate(const FLOAT *const readField, FLOAT *const writeField, const FLOAT *const rhs) const
  {


    // set pointers of 5-point stencil (only neighbour values) to very first inner grid point
    const FLOAT *readPtr_S = readField + 1;
    const FLOAT *readPtr_W = readField + (_nx + 2);
    const FLOAT *readPtr_E = readField + (_nx + 4);
    const FLOAT *readPtr_N = readField + (2 * _nx + 5);

    const FLOAT *rhsPtr = rhs + (_nx + 3);

    FLOAT *writePtr = writeField + (_nx + 3);

    // use pos to advance access through the whole grid without any expensive index computations
    unsigned int tilesize = 4;
    unsigned int pos = 0;
    #pragma omp parallel for collapse(2)
    for (unsigned int y = 1; y < _ny + 1; y+=tilesize)
    {
      for (unsigned int x = 1; x < _nx + 1; x+=tilesize)
      {

        for(unsigned int ty = y; ty < y + tilesize && ty < _ny + 1; ty++){
            for(unsigned int tx = x; tx < x + tilesize && tx < _nx + 1; tx++){
                    printf("%d ", pos);
                    writePtr[pos] = _RHS * rhsPtr[pos];
                    writePtr[pos] += _X * (readPtr_W[pos] + readPtr_E[pos]);
                    writePtr[pos] += _Y * (readPtr_S[pos] + readPtr_N[pos]);
                    pos++;
            }
            printf("\n");
            pos += tilesize +1;
        }

        // do Jacobi update and write to writePtr


        // update pos along x-axis

      }

      // update pos along y-axis; therefore just jump over the two boundary values

    }
  }

private:
  // returns the prefactor for the Jacobi stencil in x-direction
  FLOAT getX(const unsigned int nx, const unsigned int ny) const
  {
    const FLOAT hx = 1.0 / (nx + 1);
    const FLOAT hy = 1.0 / (ny + 1);
    return hy * hy / (2.0 * (hx * hx + hy * hy));
  }
  // returns the prefactor for the Jacobi stencil in y-direction
  FLOAT getY(const unsigned int nx, const unsigned int ny) const
  {
    const FLOAT hx = 1.0 / (nx + 1);
    const FLOAT hy = 1.0 / (ny + 1);
    return hx * hx / (2.0 * (hx * hx + hy * hy));
  }
  // returns the prefactor for the right hand side in Jacobi computation
  FLOAT getRHS(const unsigned int nx, const unsigned int ny) const
  {
    const FLOAT hx = 1.0 / (nx + 1);
    const FLOAT hy = 1.0 / (ny + 1);
    return -1.0 / (2.0 / hx / hx + 2.0 / hy / hy);
  }

  // number of inner grid points
  const unsigned int _nx;
  const unsigned int _ny;
  // prefactors in Jacobi computations
  const FLOAT _X;
  const FLOAT _Y;
  const FLOAT _RHS;
};

#endif // _JACOBI_H_
