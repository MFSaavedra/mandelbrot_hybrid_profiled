/*
 * mandel_dlt.cpp -- DLT retrodiction of the report-27 laptop+yeco frame
 * distribution, using DLTlib's collection-aware "image processing" solver
 * (Network::SolveImage; G. Barlas, GPL-3.0, ../../../DLTlib).
 *
 * Model mapping (all costs in seconds per frame, L = 100 frames):
 *   - distribution cost a*l*part*L : a = 0  (every node reads the same tiny
 *                                    spec.in; shipping it is a per-run
 *                                    constant, not a per-frame cost)
 *   - computation  cost p*(part*L) : p from solo-run walls / 100
 *   - collection   cost c*l*part*L : c = 1, l = measured per-frame PNG
 *                                    collection cost on the yeco WAN link
 *
 * Measured inputs (experiments/25-frame-distribution/, report 27):
 *   p_laptop = 21.02 s / 100 = 0.2102 s/frame   (laptop_hybrid baseline)
 *   p_yeco   =  5.15 s / 100 = 0.0515 s/frame   (yeco solo)
 *   l_scp    =  6.60 s / 79  = 0.0835 s/frame   (per-file scp micro-A/B)
 *   l_tar    =  0.96 s / 79  = 0.0122 s/frame   (single tar stream micro-A/B)
 *   (outlook) p_ivy = 4.56 x laptop = 0.958 s/frame, l_ivy ~ 0.004 s/frame (LAN)
 *
 * Build & run: ./run.sh (copies DLTlib to build/, patches the hard-coded
 * random.h include, compiles with -lglpk, regenerates results.txt + sweep.csv).
 *
 * Usage: ./mandel_dlt          -> the analysis table (results.txt)
 *        ./mandel_dlt sweep    -> CSV of optimum share/wall vs l (sweep.csv)
 */
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <iostream>

using namespace std;

long global_random_seed;
#include "dltlib.cpp"

const double P_LAPTOP = 0.2102;
const double P_YECO   = 0.0515;
const double L_SCP    = 0.0835;
const double L_TAR    = 0.0122;
const long   L        = 100;    // frames
const double FIXED    = 1.2;    // per-run orchestration constant (outside the model)

// Solve the 2-node problem for a given collection link cost. Returns the
// compute wall through *wall and yeco's share in frames through *nyeco.
static void solve2(const char *label, double link, double *wall = NULL,
                   double *nyeco = NULL)
{
    Network net;
    net.InsertNode((char *)"laptop", P_LAPTOP, 0, (char *)NULL, 0, true);
    net.InsertNode((char *)"yeco",   P_YECO,   0, (char *)"laptop", link, true);

    net.SolveImage(L, 0.0, 1.0);
    if (net.valid != 1) { printf("%s: no valid solution\n", label); return; }

    double part0 = net.head->part;
    double part1 = net.head->child[0]->part;
    double w     = P_LAPTOP * part0 * L;              // root computes start-to-finish
    double lane1 = (P_YECO + link) * part1 * L;       // yeco compute + collection lane
    if (wall)  { *wall = w; *nyeco = part1 * L; return; }   // sweep mode: no print
    printf("%-28s  laptop %5.1f frames | yeco %5.1f frames | "
           "wall %5.2f s (+%.1f fixed = %5.2f s)  [lanes %.2f | %.2f]\n",
           label, part0 * L, part1 * L, w, FIXED, w + FIXED, w, lane1);
}

// Evaluate a FIXED partition (our DIST_WEIGHTS choice) under a link cost.
static void evalFixed(const char *label, double n_laptop, double n_yeco, double link)
{
    double t0 = P_LAPTOP * n_laptop;
    double t1 = (P_YECO + link) * n_yeco;
    double wall = MAX(t0, t1);
    printf("%-28s  laptop %5.1f frames | yeco %5.1f frames | "
           "wall %5.2f s (+%.1f fixed = %5.2f s)  [lanes %.2f | %.2f]\n",
           label, n_laptop, n_yeco, wall, FIXED, wall + FIXED, t0, t1);
}

// 3-node closed form by hand: DLTlib's SolveImage assumes one link speed per
// parent (ImageAggregate reads temp->link[0] for every child), so the mixed
// WAN+LAN star needs the direct lane-equalization formula instead:
// equalize n_i*(p_i + l_i) with l_laptop = 0  ->  n_i proportional 1/(p_i + l_i).
static void solve3(const char *label, double p_ivy, double l_ivy, double link_yeco)
{
    double r0 = 1.0 / P_LAPTOP;
    double r1 = 1.0 / (P_YECO + link_yeco);
    double r2 = 1.0 / (p_ivy + l_ivy);
    double R  = r0 + r1 + r2;
    double n0 = L * r0 / R, n1 = L * r1 / R, n2 = L * r2 / R;
    printf("%-28s  laptop %5.1f | yeco %5.1f | ivy %5.1f frames | "
           "wall %5.2f s (+%.1f fixed = %5.2f s)\n",
           label, n0, n1, n2, n0 * P_LAPTOP, FIXED, n0 * P_LAPTOP + FIXED);
}

int main(int argc, char **argv)
{
    if (argc > 1 && !strcmp(argv[1], "sweep")) {
        // Optimum share and wall as a function of the per-frame collection
        // cost l, from the library solver at each point.
        printf("l_s_per_frame,yeco_frames,compute_wall_s\n");
        for (double link = 0.0; link <= 0.1001; link += 0.001) {
            double wall, ny;
            solve2("", link < 1e-12 ? 1e-9 : link, &wall, &ny);
            printf("%.3f,%.3f,%.4f\n", link, ny, wall);
        }
        return 0;
    }

    printf("== DLT optimum (SolveImage, a=0, c=1), 2 nodes ==\n");
    solve2("compute-only (l=0)", 1e-9);
    solve2("scp collection (l=83.5ms)", L_SCP);
    solve2("tar collection (l=12.2ms)", L_TAR);

    printf("\n== our static weights 1:4 (yeco 79 / laptop 21) evaluated ==\n");
    evalFixed("weights 1:4 under scp", 21, 79, L_SCP);
    evalFixed("weights 1:4 under tar", 21, 79, L_TAR);

    printf("\n== dynamic-stealing endpoints observed (report 27) ==\n");
    evalFixed("dyn scp: 42/58 (measured)", 42, 58, L_SCP);
    evalFixed("DLT scp optimum rounded", 39, 61, L_SCP);

    printf("\n== 3-node outlook: + ivy (p=0.958, l~4ms LAN), tar ==\n");
    solve3("laptop+yeco+ivy", 0.958, 0.004, L_TAR);

    return 0;
}
