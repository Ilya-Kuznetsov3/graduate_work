#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/physics/transformations.h>

#include <fstream>
#include <iostream>
#include <iomanip>

#define DIM 3

using namespace dealii;

struct Material {
  double p;
  double E, nu;
  double lambda, mu;

  Material(double p, double E, double nu);
};

Material::Material(double p, double E, double nu)
  : p(p), E(E), nu(nu) {
  lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
  mu = E / (2 * (1 + nu));
}

SymmetricTensor<4, DIM> get_stress_tensor(Material& material) {
  SymmetricTensor<4, DIM> stress_tensor;
    for (unsigned int i = 0; i < DIM; ++i)
      for (unsigned int j = 0; j < DIM; ++j)
        for (unsigned int k = 0; k < DIM; ++k)
          for (unsigned int l = 0; l < DIM; ++l)
            stress_tensor[i][j][k][l] = (((i == k) && (j == l) ? material.mu : 0.0) +
                               ((i == l) && (j == k) ? material.mu : 0.0) +
                               ((i == j) && (k == l) ? material.lambda : 0.0));
    return stress_tensor;

}


SymmetricTensor<2, DIM> get_strain_tensor(const FEValues<DIM> &fe_values,
                                   const unsigned int   shape_func,
                                   const unsigned int   q_point) {
  SymmetricTensor<2, DIM> strain_tensor;

  for (unsigned int i = 0; i < DIM; ++i)
    strain_tensor[i][i] = fe_values.shape_grad_component(shape_func, q_point, i)[i];
  
  for (unsigned int i = 0; i < DIM; ++i)
    for (unsigned int j = i + 1; j < DIM; ++j)
      strain_tensor[i][j] =
        (fe_values.shape_grad_component(shape_func, q_point, i)[j] +
         fe_values.shape_grad_component(shape_func, q_point, j)[i]) / 2.0;
  
  return strain_tensor;
}

SymmetricTensor<2, DIM> get_strain_tensor(const std::vector<Tensor<1, DIM>> &grad) {
  SymmetricTensor<2, DIM> strain_tensor;

  for (unsigned int i = 0; i < DIM; ++i)
    strain_tensor[i][i] = grad[i][i];

  for (unsigned int i = 0; i < DIM; ++i)
    for (unsigned int j = i + 1; j < DIM; ++j)
      strain_tensor[i][j] = (grad[i][j] + grad[j][i]) / 2.0;

  return strain_tensor;
}

Tensor<2, 3> get_rotation_matrix(const std::vector<Tensor<1, 3>> &grad_u) {
  const Tensor<1, 3> curl({grad_u[2][1] - grad_u[1][2],
                           grad_u[0][2] - grad_u[2][0],
                           grad_u[1][0] - grad_u[0][1]});
  const double tan_angle = std::sqrt(curl * curl);
  const double angle     = std::atan(tan_angle);

  if (std::abs(angle) < 1e-9) {
    static const double id[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    static const Tensor<2, 3> rot(id);
    return rot;
  }

  const Tensor<1, 3> axis = curl / tan_angle;
  return Physics::Transformations::Rotations::rotation_matrix_3d(axis,
                                                                   -angle);
}


class TopBoundaryValues : public Function<DIM> {
public:
  TopBoundaryValues(double timestep);
  void vector_value(const Point<DIM>&, Vector<double>&) const override;
  void vector_value_list(const std::vector<Point<DIM>> &points,
                         std::vector<Vector<double>> &value_list) const override;
private:
  const double velocity;
  const double timestep;
};

TopBoundaryValues::TopBoundaryValues(double timestep)
    : Function<DIM>(DIM), velocity(.08), timestep(timestep)  {}


void TopBoundaryValues::vector_value(const Point<DIM>&,
                                     Vector<double>& values) const {
    values.reinit(DIM);
    values = 0;
    values(DIM-1) = -timestep * velocity;
  }


  void TopBoundaryValues::vector_value_list(const std::vector<Point<DIM>>& pts,
                                                         std::vector<Vector<double>>& list) const {
    for (unsigned int i = 0; i < pts.size(); ++i)
      vector_value(pts[i], list[i]);
  }

class BodyForce : public Function<DIM> {
public:
  BodyForce() : Function<DIM>(DIM) {}
  void vector_value(const Point<DIM>&, Vector<double>&) const override;
  void vector_value_list(const std::vector<Point<DIM>>&,
                         std::vector<Vector<double>>&) const override;
};

void BodyForce::vector_value(const Point<DIM>&, Vector<double> &values) const {
  const double g  = 9.81, rho = 7850;
  values.reinit(DIM);
  values = 0;
  values(DIM - 1) = -rho * g;
}

void BodyForce::vector_value_list(const std::vector<Point<DIM>>& pts,
                                       std::vector<Vector<double>>& v_list) const {
  for (unsigned int i = 0; i < pts.size(); ++i)
    vector_value(pts[i], v_list[i]);
}

class History {
public:
  Tensor<1, DIM> dot_u;
  Tensor<1, DIM> dot_dot_u;
  SymmetricTensor<2, DIM> prev_stress;

  constexpr static double beta = 0.25;
  constexpr static double gamma = 0.5;
};

class System {
public:
  System(Material material, double end_time, double timestep);
  void solve();
private:
  void solve_step();
  void create_grid();
  void setup_system();
  void assemble_system();
  void solve_system();
  void move_mesh();
  void setup_history();
  void update_history();
  void output_results();


  const Material material;

  double present_time;
  double end_time;
  double timestep;
  unsigned int step;

  parallel::shared::Triangulation<DIM> triangulation;
  const FESystem<DIM> fe;
  DoFHandler<DIM> dof_handler;
  AffineConstraints<double> hanging_node_constraints;
  const QGauss<DIM> quadrature_formula;

  std::vector<History> history;

  PETScWrappers::MPI::SparseMatrix system_matrix;
  PETScWrappers::MPI::Vector system_rhs;
  Vector<double> incremental_displacement;
  SymmetricTensor<4, DIM> stress_tensor;

  MPI_Comm mpi_communicator;

  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;
};

System::System(Material material, double end_time, double timestep)
  : material(material),
    present_time(0.0),
    end_time(end_time),
    timestep(timestep),
    step(0),
    triangulation(MPI_COMM_WORLD), 
    fe(FE_Q<DIM>(1) ^ DIM), 
    dof_handler(triangulation), 
    quadrature_formula(fe.degree + 1), 
    mpi_communicator(MPI_COMM_WORLD), 
    n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)), 
    this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
     {
      stress_tensor = get_stress_tensor(material);
    }

void System::create_grid() {
  const double inner_radius = 0.8, outer_radius = 1.0;
  GridGenerator::cylinder_shell(triangulation, 3, inner_radius, outer_radius);

  for (const auto &cell : triangulation.active_cell_iterators())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary()) {
        const Point<DIM> fc = face->center();
        if (std::abs(fc[2]) < 1e-12)
          face->set_boundary_id(0);
        else if (std::abs(fc[2] - 3.0) < 1e-12)
          face->set_boundary_id(1);
        else if (fc[0]*fc[0] + fc[1]*fc[1] < std::pow((inner_radius+outer_radius)/2,2))
            face->set_boundary_id(2);
        else
            face->set_boundary_id(3);
      }
  triangulation.refine_global(2);
}

void System::setup_system() {
  dof_handler.distribute_dofs(fe);

  locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  hanging_node_constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, hanging_node_constraints);
  hanging_node_constraints.close();

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, hanging_node_constraints, false);
    
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs,
                                             mpi_communicator,
                                             locally_relevant_dofs);
  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       mpi_communicator);

  system_rhs.reinit(locally_owned_dofs, mpi_communicator);

  incremental_displacement.reinit(dof_handler.n_dofs());
}

  void System::assemble_system() {
    system_matrix = 0;
    system_rhs    = 0;

    FEValues<DIM> fe_values(fe, quadrature_formula,
                            update_values     | update_gradients |
                            update_quadrature_points   | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    BodyForce body_force;
    std::vector<Vector<double>> body_force_vals(n_q_points, Vector<double>(DIM));

    double lhs_factor = material.p / (History::beta * timestep * timestep);
    double velocity_factor = material.p / (History::beta * timestep);
    double accel_factor = (material.p * (1 - 2 * History::beta)) / (2 * History::beta);

    for (auto &cell : dof_handler.active_cell_iterators()) {
      if (cell->is_locally_owned()) {
        cell_matrix = 0;
        cell_rhs = 0;
        fe_values.reinit(cell);

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const unsigned int comp_i = fe.system_to_component_index(i).first;
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            const unsigned int comp_j = fe.system_to_component_index(j).first;
            for (unsigned int q = 0; q < n_q_points; ++q) {
              const SymmetricTensor<2,DIM> eps_i = get_strain_tensor(fe_values,i,q);
              const SymmetricTensor<2,DIM> eps_j = get_strain_tensor(fe_values,j,q);

              cell_matrix(i,j) -= (eps_i * stress_tensor * eps_j) * fe_values.JxW(q);
              
              if (comp_i == comp_j) {
                cell_matrix(i,j) -= lhs_factor * fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * fe_values.JxW(q);
              }
            }
          }
        }

        const History* local_qph = reinterpret_cast<History*>(cell->user_pointer());
        body_force.vector_value_list(fe_values.get_quadrature_points(), body_force_vals);

        for (unsigned int i=0;i<dofs_per_cell;++i) {
          const unsigned int comp_i = fe.system_to_component_index(i).first;
          for (unsigned int q=0; q < n_q_points; ++q) {
            cell_rhs(i) += (- body_force_vals[q](comp_i) * fe_values.shape_value(i,q)
                            + local_qph[q].prev_stress * get_strain_tensor(fe_values,i,q)
                            + velocity_factor * local_qph[q].dot_u[comp_i] * fe_values.shape_value(i,q) 
                            + accel_factor * local_qph[q].dot_dot_u[comp_i] * fe_values.shape_value(i,q)) * fe_values.JxW(q);

          }
        }

        cell->get_dof_indices(local_dof_indices);
        hanging_node_constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                                            local_dof_indices,
                                                            system_matrix, system_rhs);
    }
  }
  
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  const FEValuesExtractors::Scalar z_component(DIM-1);

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,0,
                                             Functions::ZeroFunction<DIM>(DIM),
                                             boundary_values);

  VectorTools::interpolate_boundary_values(dof_handler, 1,
    TopBoundaryValues(timestep),
    boundary_values, fe.component_mask(z_component));

  PETScWrappers::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);
  MatrixTools::apply_boundary_values(
    boundary_values, system_matrix, tmp, system_rhs, false);
  incremental_displacement = tmp;

  }

void System::solve_system() {
  std::cout << "solve" << std::endl;

  PETScWrappers::MPI::Vector distributed_incremental_displacement(
  locally_owned_dofs, mpi_communicator);
  distributed_incremental_displacement = incremental_displacement;

  SolverControl solver_control(dof_handler.n_dofs(), 1e-16 * system_rhs.l2_norm());
  PETScWrappers::SolverCG solver(solver_control);
  PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);

  solver.solve(system_matrix, distributed_incremental_displacement, system_rhs, preconditioner);
  incremental_displacement = distributed_incremental_displacement;
  std::cout << "solved" << std::endl;
  hanging_node_constraints.distribute(incremental_displacement);
}

void System::move_mesh() {
  std::vector<bool> shifted_vertex(triangulation.n_vertices(), false);

  for (auto &cell : dof_handler.active_cell_iterators())
    for (const auto vertex : cell->vertex_indices())
      if (!shifted_vertex[cell->vertex_index(vertex)]) {
        shifted_vertex[cell->vertex_index(vertex)] = true;
        Point<DIM> shift;
        for (unsigned int d = 0; d < DIM; ++d)
          shift[d] = incremental_displacement(cell->vertex_dof_index(vertex, d));
        cell->vertex(vertex) += shift;
      }

}

void System::output_results()  {
  DataOut<DIM> data_out;
  data_out.attach_dof_handler(dof_handler);

  std::vector<std::string> names = {"delta_x","delta_y","delta_z"};

  data_out.add_data_vector(incremental_displacement, names);

  Vector<double> norm_of_stress(triangulation.n_active_cells());
  for (const auto &cell : triangulation.active_cell_iterators()) {
    SymmetricTensor<2,DIM> acc_stress;

    for (unsigned int q=0;q<quadrature_formula.size();++q)
      acc_stress += reinterpret_cast<History*>(cell->user_pointer())[q].prev_stress;
    norm_of_stress(cell->active_cell_index()) =
      (acc_stress / quadrature_formula.size()).norm();
    }

    data_out.add_data_vector(norm_of_stress, "norm_of_stress");

    data_out.build_patches();
    std::ofstream out("solution-" + Utilities::int_to_string(step ,4) + ".vtu");
    data_out.write_vtu(out);
    
}

void System::setup_history() {
  triangulation.clear_user_data();
  history.clear();
  history.resize(triangulation.n_locally_owned_active_cells() * quadrature_formula.size());
  unsigned int hist_idx = 0;

  for (auto &cell : triangulation.active_cell_iterators()) {
    if (cell->is_locally_owned()) {
      cell->set_user_pointer(&history[hist_idx]);

      for (unsigned int q = 0; q < quadrature_formula.size(); ++q) {
          auto &point_history = history[hist_idx + q];
          point_history.dot_u = Tensor<1,DIM>();
          point_history.dot_dot_u = Tensor<1,DIM>();  
        }

      hist_idx += quadrature_formula.size();
    }
  }
}


void System::update_history() {

    FEValues<DIM> fe_values(fe, quadrature_formula, update_values | update_quadrature_points | update_gradients);
    const FEValuesExtractors::Vector disp(0); 
    std::vector<std::vector<Tensor<1,DIM>>> grad_u(quadrature_formula.size(), std::vector<Tensor<1,DIM>>(DIM));
    std::vector<Tensor<1,DIM>> du_q(quadrature_formula.size());

    for (auto &cell : dof_handler.active_cell_iterators()) {
      if (cell->is_locally_owned()) {
        History* qph = reinterpret_cast<History*>(cell->user_pointer());
        fe_values.reinit(cell);
        fe_values.get_function_gradients(incremental_displacement, grad_u);
        fe_values[disp].get_function_values(incremental_displacement, du_q);

        for (unsigned int q=0;q<quadrature_formula.size();++q) {

          const SymmetricTensor<2,DIM> new_stress = qph[q].prev_stress +
            (stress_tensor * get_strain_tensor(grad_u[q]));
          const Tensor<2,DIM> R = get_rotation_matrix(grad_u[q]);
          qph[q].prev_stress = symmetrize(transpose(R) * Tensor<2,DIM>(new_stress) * R);

          const Tensor<1,DIM> dot_dot_u_n =
                (1.0 / (History::beta * timestep * timestep)) * du_q[q]
              - (1.0 / (History::beta * timestep)) * qph[q].dot_u
              - ((1.0 - 2.0 * History::beta) / (2.0 * History::beta)) * qph[q].dot_dot_u;


          const Tensor<1,DIM> dot_u_n =
                qph[q].dot_u + timestep * ( (1.0 - History::gamma) * qph[q].dot_dot_u + History::gamma * dot_dot_u_n );

          qph[q].dot_u = dot_u_n;
          qph[q].dot_dot_u = dot_dot_u_n;
        }
    }
  }
}

void System::solve_step() {
  assemble_system();
  solve_system();
  update_history();
  move_mesh();
 // output_results();
}

void System::solve() {
  create_grid();
  setup_history();
  setup_system();
  while (present_time < end_time) {
    std::cout << "Step " << present_time << std::endl;
    solve_step();
    present_time += timestep;
    ++step;
  }
}

int main(int argc, char **argv) {
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Material material(7850, 200e10, 0.3);
  std::cout << "Start" << std::endl;
  System system(material, 10, 1);
  system.solve();
  return 0;
}