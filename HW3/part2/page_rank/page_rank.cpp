#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  
  #pragma omp parallel for
  for (int i = 0; i < numNodes; ++i)
  {
    solution[i] = equal_prob;
  }

  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }
   */

  bool converged = false;
  double *old_solution = (double*)malloc(sizeof(double)*numNodes);
  double *tmp_global = (double*)malloc(sizeof(double)*numNodes);

  #pragma omp parallel for
  for (int i = 0; i < numNodes; ++i)
  {
    old_solution[i] = equal_prob;
  }

  double no_out = 0.0;
  double dam_num = damping/numNodes;


  // printf("%f\n", equal_prob);
  while(!converged)
  {
    #pragma omp parallel for
    for (int i = 0; i < numNodes; ++i)
    {
      old_solution[i] = solution[i];
      solution[i] = 0;
    }
    double global_diff = 0.0;

    for (int v = 0; v < numNodes; v++)
    {
      if (outgoing_size(g, v) == 0)
      {
         no_out += old_solution[v];
      }
    }
    no_out *= dam_num;

    #pragma omp parallel for
    for (int i = 0; i < numNodes; ++i)
    {
      const Vertex* start = incoming_begin(g, i);
      const Vertex* end = incoming_end(g, i);

      // #pragma omp parallel for 
      for (const Vertex* v=start; v!=end; v++)
      {
        solution[i] += old_solution[*v]/outgoing_size(g, *v);
      }

      solution[i] = (damping * solution[i]) + (1.0-damping) / numNodes;
      // for (int v = 0; v < numNodes; v++)
      // {
      //   if (outgoing_size(g, v) == 0)
      //   {
      //     solution[i] += damping*old_solution[v]/numNodes;
      //   }
      // }
      solution[i]+=no_out;
      // global_diff += abs(solution[i] - old_solution[i]);

      // tmp_global[i] = abs(solution[i] - old_solution[i]);
      tmp_global[i] = solution[i] - old_solution[i];
      if (tmp_global[i] < 0)
      {
        tmp_global[i] = -tmp_global[i];
      }
    }
    for (int i = 0; i < numNodes; ++i)
    {
      global_diff += tmp_global[i];
    }

    converged = (global_diff < convergence);
    // printf("%f\n", global_diff);
  }

  free(old_solution);

}
