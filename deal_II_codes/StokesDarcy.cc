/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2011 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth, Texas A&M University, 2011
 */


// @sect3{Include files}

// The include files for this program are the same as for many others
// before. The only new one is the one that declares FE_Nothing as discussed
// in the introduction. The ones in the hp directory have already been
// discussed in step-27.

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_raviart_thomas.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <iostream>
#include <fstream>
#include <sstream>


namespace StokesDarcy
{
  using namespace dealii;

  template <int dim>
  class FluidStructureProblem
  {
  public:
    FluidStructureProblem (const unsigned int stokes_degree,
                           const unsigned int darcy_degree);
    void run ();

  private:
    enum
    {
      fluid_domain_id,
      solid_domain_id
    };

    static bool
    cell_is_in_fluid_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell);

    static bool
    cell_is_in_solid_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell);


    void make_grid ();
    void set_active_fe_indices ();
    void setup_dofs ();
    void assemble_system ();
    void assemble_fluid_interface_term (const FEFaceValuesBase<dim>          &darcy_fe_face_values,
                                        const FEFaceValuesBase<dim>          &stokes_fe_face_values,
                                        std::vector<Tensor<1,dim> >          &lm_phi,
                                        std::vector<Tensor<1,dim> >          &stokes_phi_u,
                                        FullMatrix<double>                   &local_fluid_interface_matrix) const;
    void assemble_fluid_BJS_interface_term (const FEFaceValuesBase<dim>      &stokes_fe_face_values,
                                            std::vector<Tensor<1,dim> >      &stokes_phi_u,
                                            FullMatrix<double>               &local_fluid_BJS_interface_matrix) const;
    void assemble_solid_interface_term (const FEFaceValuesBase<dim>          &darcy_fe_face_values,
                                        std::vector<Tensor<1,dim> >          &lm_phi,
                                        std::vector<Tensor<1,dim> >          &stokes_phi_u,
                                        FullMatrix<double>                   &local_solid_interface_matrix) const;
    void solve ();
    void output_results () const;

    const unsigned int    stokes_degree;
    const unsigned int    darcy_degree;

    Triangulation<dim>    triangulation;
    FESystem<dim>         stokes_fe;
    FESystem<dim>         darcy_fe;
    hp::FECollection<dim> fe_collection;
    hp::DoFHandler<dim>   dof_handler;

    ConstraintMatrix      constraints;

    SparsityPattern       sparsity_pattern;
    SparseMatrix<double>  system_matrix;

    Vector<double>        solution;
    Vector<double>        system_rhs;

    const double viscosity;
    const double alpha_bjs;


  };


  template <int dim>
  class StokesBoundaryValues : public Function<dim>
  {
  public:
    StokesBoundaryValues () : Function<dim>(3*dim+2) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;
  };


  template <int dim>
  double
  StokesBoundaryValues<dim>::value (const Point<dim>  &p,
                                    const unsigned int component) const
  {
    Assert (component < this->n_components,
            ExcIndexRange (component, 0, this->n_components));

    if (component == 0)
      switch (dim)
        {
        case 2:
          return std::sin(p[0]/(2*std::sqrt(0.1))+1.05)*std::exp(p[1]/(2*std::sqrt(0.1)));
          // return (1-p[0])*(1-p[0])*(1-p[1])*(1-p[1]);
          //  return 0;
        case 3:
          return 0;
        default:
          Assert (false, ExcNotImplemented());
        }

    if (component == 1)
      switch (dim)
        {
        case 2:
          return -std::cos(p[0]/(2*std::sqrt(0.1))+1.05)*std::exp(p[1]/(2*std::sqrt(0.1)));
          // return 0;
        case 3:
          return 0;
        default:
          Assert (false, ExcNotImplemented());
        }

    return 0;
  }


  template <int dim>
  void
  StokesBoundaryValues<dim>::vector_value (const Point<dim> &p,
                                           Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = StokesBoundaryValues<dim>::value (p, c);
  }


  template <int dim>
  class DarcyBoundaryValues : public Function<dim>
  {
  public:
    DarcyBoundaryValues () : Function<dim>(3*dim+2) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };


  template <int dim>
  double
  DarcyBoundaryValues<dim>::value (const Point<dim>  &p,
                                   const unsigned int component) const
  {
    Assert (component < this->n_components,
            ExcIndexRange (component, 0, this->n_components));

    return (std::sqrt(0.1)/2)*std::cos(p[0]/(2*std::sqrt(0.1))+1.05)*std::exp(p[1]/(2*std::sqrt(0.1)));
    // return 0;
  }


  template <int dim>
  class RightHandSideStokesSource : public Function<dim>
  {
  public:
    RightHandSideStokesSource () : Function<dim>(dim+1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;

  };


  template <int dim>
  double
  RightHandSideStokesSource<dim>::value (const Point<dim>  &p,
                                         const unsigned int component) const
  {
    if (component == 0)
      switch (dim)
        {
        case 2:
          return (0.1*0.5/std::sqrt(0.1)-1)*std::sin(p[0]/(2*std::sqrt(0.1))+1.05)*std::exp(1/(4*std::sqrt(0.1)))+p[1]-0.5;
        case 3:
          return 0;
        default:
          Assert (false, ExcNotImplemented());
        }

    if (component == 1)
      switch (dim)
        {
        case 2:
          return 1;
        case 3:
          return 0;
        default:
          Assert (false, ExcNotImplemented());
        }

    return 0;
  }


  template <int dim>
  void
  RightHandSideStokesSource<dim>::vector_value (const Point<dim> &p,
                                                Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = RightHandSideStokesSource<dim>::value (p, c);
  }

  template <int dim>
  class RightHandSideDarcySource : public Function<dim>
  {
  public:
    RightHandSideDarcySource () : Function<dim>(dim+1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &value) const;

  };


  template <int dim>
  double
  RightHandSideDarcySource<dim>::value (const Point<dim>  &p,
                                        const unsigned int component) const
  {
    if (component == 0)
      switch (dim)
        {
        case 2:
          return -0.9*std::sin(p[0]/(2*std::sqrt(0.1))+1.05)*std::exp(p[1]/(2*std::sqrt(0.1)));
        case 3:
          return 0;
        default:
          Assert (false, ExcNotImplemented());
        }

    if (component == 1)
      switch (dim)
        {
        case 2:
          return 0.9*std::cos(p[0]/(2*std::sqrt(0.1))+1.05)*std::exp(p[1]/(2*std::sqrt(0.1)));
        case 3:
          return 0;
        default:
          Assert (false, ExcNotImplemented());
        }

    return 0;
  }


  template <int dim>
  void
  RightHandSideDarcySource<dim>::vector_value (const Point<dim> &p,
                                               Vector<double>   &values) const
  {
    for (unsigned int c=0; c<this->n_components; ++c)
      values(c) = RightHandSideDarcySource<dim>::value (p, c);
  }

  template <int dim>
  class KInverse : public TensorFunction<2,dim>
  {
  public:
    KInverse () : TensorFunction<2,dim>() {}
    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const;
  };
  template <int dim>
  void
  KInverse<dim>::value_list (const std::vector<Point<dim> > &points,
                             std::vector<Tensor<2,dim> >    &values) const
  {
    Assert (points.size() == values.size(),
            ExcDimensionMismatch (points.size(), values.size()));
    for (unsigned int p=0; p<points.size(); ++p)
      {
        values[p].clear ();
        for (unsigned int d=0; d<dim; ++d)
          values[p][d][d] = 1.;
      }
  }


  template <int dim>
  FluidStructureProblem<dim>::
  FluidStructureProblem (const unsigned int stokes_degree,
                         const unsigned int darcy_degree)
    :
      stokes_degree (stokes_degree),
      darcy_degree (darcy_degree),
      triangulation (Triangulation<dim>::maximum_smoothing),
      stokes_fe (FE_Q<dim>(stokes_degree+1), dim,
                 FE_Q<dim>(stokes_degree), 1,
                 FE_Nothing<dim>(), dim,
                 FE_Nothing<dim>(), 1,
                 FE_Nothing<dim>(), dim),
      darcy_fe (FE_Nothing<dim>(), dim,
                FE_Nothing<dim>(), 1,
                FE_Q<dim>(darcy_degree), dim,
                FE_Q<dim>(darcy_degree),1,
                FE_Q<dim>(darcy_degree), dim),
      dof_handler (triangulation),
      viscosity (0.1),
      alpha_bjs (1)
  {
    fe_collection.push_back (stokes_fe);
    fe_collection.push_back (darcy_fe);
  }




  template <int dim>
  bool
  FluidStructureProblem<dim>::
  cell_is_in_fluid_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell)
  {
    return (cell->material_id() == fluid_domain_id);
  }


  template <int dim>
  bool
  FluidStructureProblem<dim>::
  cell_is_in_solid_domain (const typename hp::DoFHandler<dim>::cell_iterator &cell)
  {
    return (cell->material_id() == solid_domain_id);
  }


  template <int dim>
  void
  FluidStructureProblem<dim>::make_grid ()
  {
    Point<dim> p1,p2;
    p1[0] = 0;
    p1[1] = 0;
    p2[0] = 1;
    p2[1] = 1;
    std::vector<unsigned int> vec ={32,32};
    GridGenerator::subdivided_hyper_rectangle (triangulation, vec,p1,p2);

    for (typename Triangulation<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
      if (std::fabs(cell->center()[1]) > 0.5)
        cell->set_material_id (fluid_domain_id);
      else
        cell->set_material_id (solid_domain_id);

    for (typename Triangulation<dim>::active_cell_iterator
         cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell)
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        {
          if ((cell->face(f)->at_boundary())
              &&
              (cell->face(f)->center()[1]>0.5))
            cell->face(f)->set_all_boundary_ids(1);
          if ((cell->face(f)->at_boundary())
              &&
              (cell->face(f)->center()[1] >0.5)
              &&
              (cell->face(f)->center()[0] ==1))
            cell->face(f)->set_all_boundary_ids(3);
          if ((cell->face(f)->at_boundary())
              &&
              (cell->face(f)->center()[1] <0.5))
            cell->face(f)->set_all_boundary_ids(2);
        }
  }


  template <int dim>
  void
  FluidStructureProblem<dim>::set_active_fe_indices ()
  {
    for (typename hp::DoFHandler<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
      {
        if (cell_is_in_fluid_domain(cell))
          cell->set_active_fe_index (0);
        else if (cell_is_in_solid_domain(cell))
          cell->set_active_fe_index (1);
        else
          Assert (false, ExcNotImplemented());
      }
  }


  template <int dim>
  void
  FluidStructureProblem<dim>::setup_dofs ()
  {
    set_active_fe_indices ();
    dof_handler.distribute_dofs (fe_collection);
    DoFRenumbering::component_wise(dof_handler);


    {
      const FEValuesExtractors::Vector stokes_velocities(0);

      std::map<types::global_dof_index,double> boundary_values;

    }



    std::cout << "   Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl
              << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;

    {
      DynamicSparsityPattern dsp (dof_handler.n_dofs(),
                                  dof_handler.n_dofs());

      Table<2,DoFTools::Coupling> cell_coupling (fe_collection.n_components(),
                                                 fe_collection.n_components());
      Table<2,DoFTools::Coupling> face_coupling (fe_collection.n_components(),
                                                 fe_collection.n_components());

      for (unsigned int c=0; c<fe_collection.n_components(); ++c)
        for (unsigned int d=0; d<fe_collection.n_components(); ++d)
          {
            if (((c<dim+1) && (d<dim+1)
                 && !((c==dim) && (d==dim)))
                ||
                ((c>=dim+1) && (d>=dim+1)))
              cell_coupling[c][d] = DoFTools::always;

            if ((c>=dim+1) && (d<dim+1))
              face_coupling[c][d] = DoFTools::always;
          }

      DoFTools::make_flux_sparsity_pattern (dof_handler, dsp,
                                            cell_coupling, face_coupling);
      sparsity_pattern.copy_from (dsp);
    }

    system_matrix.reinit (sparsity_pattern);

    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
  }


  template <int dim>
  void FluidStructureProblem<dim>::assemble_system ()
  {
    system_matrix=0;
    system_rhs=0;

    const QGauss<dim> stokes_quadrature(stokes_degree+2);
    const QGauss<dim> darcy_quadrature(darcy_degree+2);

    hp::QCollection<dim>  q_collection;
    q_collection.push_back (stokes_quadrature);
    q_collection.push_back (darcy_quadrature);

    hp::FEValues<dim> hp_fe_values (fe_collection, q_collection,
                                    update_values    |
                                    update_quadrature_points  |
                                    update_JxW_values |
                                    update_gradients);

    const QGauss<dim-1> common_face_quadrature(std::max (stokes_degree+2,
                                                         darcy_degree+2));

    FEFaceValues<dim>    stokes_fe_face_values (stokes_fe,
                                                common_face_quadrature,
                                                update_values|
                                                update_JxW_values |
                                                update_normal_vectors |
                                                update_gradients);
    FEFaceValues<dim>    darcy_fe_face_values (darcy_fe,
                                               common_face_quadrature,
                                               update_values|
                                               update_JxW_values |
                                               update_quadrature_points |
                                               update_normal_vectors |
                                               update_gradients);

    FESubfaceValues<dim> stokes_fe_subface_values (stokes_fe,
                                                   common_face_quadrature,
                                                   update_JxW_values |
                                                   update_normal_vectors |
                                                   update_gradients);
    FESubfaceValues<dim> darcy_fe_subface_values (darcy_fe,
                                                  common_face_quadrature,
                                                  update_values);


    const unsigned int        stokes_dofs_per_cell = stokes_fe.dofs_per_cell;
    const unsigned int        darcy_dofs_per_cell  = darcy_fe.dofs_per_cell;


    FullMatrix<double>        local_matrix;
    FullMatrix<double>        local_fluid_interface_matrix (stokes_dofs_per_cell,
                                                            darcy_dofs_per_cell);
    FullMatrix<double>        local_solid_interface_matrix (darcy_dofs_per_cell,
                                                            darcy_dofs_per_cell);
    FullMatrix<double>        local_fluid_BJS_interface_matrix(stokes_dofs_per_cell,
                                                               stokes_dofs_per_cell);
    Vector<double>            local_rhs(darcy_dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<types::global_dof_index> neighbor_solid_dof_indices (darcy_dofs_per_cell);


    const RightHandSideStokesSource<dim>  right_hand_side_stokes_source;
    const RightHandSideDarcySource<dim>   right_hand_side_darcy_source;
    const DarcyBoundaryValues<dim>  darcy_bc;
    const KInverse<dim>              k_inverse;


    const FEValuesExtractors::Vector     stokes_velocities (0);
    const FEValuesExtractors::Scalar     stokes_pressure (dim);
    const FEValuesExtractors::Vector     darcy_velocities (dim+1);
    const FEValuesExtractors::Scalar     darcy_pressure (2*dim+1);
    const FEValuesExtractors::Vector     l_multiplier(3*dim);

    std::vector<SymmetricTensor<2,dim> > stokes_symgrad_phi_u (stokes_dofs_per_cell);
    std::vector<Tensor<1,dim> >          stokes_phi_u (stokes_dofs_per_cell);
    std::vector<double>                  stokes_div_phi_u     (stokes_dofs_per_cell);
    std::vector<double>                  stokes_phi_p         (stokes_dofs_per_cell);

    std::vector<Tensor<1,dim> >          darcy_phi_u     (darcy_dofs_per_cell);
    std::vector<double>                  darcy_div_phi_u (darcy_dofs_per_cell);
    std::vector<double>                  darcy_phi_p     (darcy_dofs_per_cell);
    std::vector<Tensor<1,dim> >          darcy_grad_phi_p     (darcy_dofs_per_cell);

    std::vector<Tensor<1,dim> >          lm_phi     (darcy_dofs_per_cell);

    const unsigned int n_face_quadrature_points= darcy_fe_face_values.n_quadrature_points;

    std::vector<double>                  darcy_bc_values (n_face_quadrature_points);


    typename hp::DoFHandler<dim>::active_cell_iterator
        cell = dof_handler.begin_active(),
        endc = dof_handler.end();
    for (; cell!=endc; ++cell)
      {
        hp_fe_values.reinit (cell);

        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

        local_matrix.reinit (cell->get_fe().dofs_per_cell,
                             cell->get_fe().dofs_per_cell);
        local_rhs.reinit (cell->get_fe().dofs_per_cell);


        if (cell_is_in_fluid_domain (cell))
          {
            const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
            std::vector< Vector<double> > rhs_stokes_source_values (fe_values.n_quadrature_points, Vector<double>(3*dim+2));
            right_hand_side_stokes_source.vector_value_list (fe_values.get_quadrature_points(),
                                        rhs_stokes_source_values);
            Assert (dofs_per_cell == stokes_dofs_per_cell,
                    ExcInternalError());

            for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
              {
                for (unsigned int k=0; k<dofs_per_cell; ++k)
                  {
                    stokes_symgrad_phi_u[k] = fe_values[stokes_velocities].symmetric_gradient (k, q);
                    stokes_phi_u[k]         = fe_values[stokes_velocities].value (k, q);
                    stokes_div_phi_u[k]     = fe_values[stokes_velocities].divergence (k, q);
                    stokes_phi_p[k]         = fe_values[stokes_pressure].value (k, q);
                  }

                for (unsigned int i=0; i<dofs_per_cell; ++i)
                  {
                    const unsigned int component_i =
                      stokes_fe.system_to_component_index(i).first;
                    local_rhs(i) += (rhs_stokes_source_values[q](component_i)*fe_values.shape_value(i,q))
                                      * fe_values.JxW(q);
                    for (unsigned int j=0; j<dofs_per_cell; ++j){
                      local_matrix(i,j) += (2 *viscosity* stokes_symgrad_phi_u[i] * stokes_symgrad_phi_u[j]
                                            - stokes_div_phi_u[i] * stokes_phi_p[j]
                                            - stokes_phi_p[i] * stokes_div_phi_u[j])
                          * fe_values.JxW(q);

                    }
                  }

          }
          }
        else
          {
            const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
            std::vector<Tensor<2,dim> >          k_inverse_values (fe_values.n_quadrature_points);
            std::vector< Vector<double> > rhs_darcy_source_values (fe_values.n_quadrature_points, Vector<double>(3*dim+2));
            Assert (dofs_per_cell == darcy_dofs_per_cell,
                    ExcInternalError());
            k_inverse.value_list (fe_values.get_quadrature_points(),
                                  k_inverse_values);
            right_hand_side_darcy_source.vector_value_list (fe_values.get_quadrature_points(),
                                        rhs_darcy_source_values);

            for (unsigned int q=0; q<fe_values.n_quadrature_points; ++q)
              {
                for (unsigned int k=0; k<dofs_per_cell; ++k)
                  {
                    darcy_phi_u[k]     = fe_values[darcy_velocities].value(k, q);
                    darcy_div_phi_u[k] = fe_values[darcy_velocities].divergence (k, q);
                    darcy_phi_p[k]     = fe_values[darcy_pressure].value (k, q);
                    darcy_grad_phi_p[k]= fe_values[darcy_pressure].gradient (k, q);
                    lm_phi[k] = fe_values[l_multiplier].value(k,q);
                  }
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                  {
                    const unsigned int component_i =
                      darcy_fe.system_to_component_index(i).first;
                    local_rhs(i) += (rhs_darcy_source_values[q](component_i)*fe_values.shape_value(i,q))
                                      * fe_values.JxW(q);
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                      {
                        local_matrix(i,j) += (k_inverse_values[q]*darcy_phi_u[i] * darcy_phi_u[j]
                                              - darcy_div_phi_u[i] * darcy_phi_p[j]
                                              - darcy_phi_p[i] * darcy_div_phi_u[j]
                                              +0.01*darcy_grad_phi_p[i]*darcy_grad_phi_p[j]
                                              *(triangulation.last()->diameter()/std::sqrt(dim))
                                              *(triangulation.last()->diameter()/std::sqrt(dim))
                                              +0.01*lm_phi[i]*lm_phi[j]*(triangulation.last()->diameter()/std::sqrt(dim))
                                              *(triangulation.last()->diameter()/std::sqrt(dim)))
                            * fe_values.JxW(q);



                      }
                  }
              }


            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
              if (cell->at_boundary(f))
                {
                  darcy_fe_face_values.reinit (cell, f);
                  darcy_bc.value_list (darcy_fe_face_values.get_quadrature_points(),darcy_bc_values);
                  for (unsigned int q=0; q<n_face_quadrature_points; ++q)
                    {
                      const Tensor<1,dim> normal_vector = darcy_fe_face_values.normal_vector(q);
                      for (unsigned int i=0; i<cell->get_fe().dofs_per_cell; ++i){
                          local_rhs(i) += -(darcy_fe_face_values[darcy_velocities].value (i, q) *
                                            normal_vector*
                                            darcy_bc_values[q] *
                                            darcy_fe_face_values.JxW(q));
                        }
                    }
                }

          }


        local_dof_indices.resize (cell->get_fe().dofs_per_cell);
        cell->get_dof_indices (local_dof_indices);

        for (unsigned int i=0; i<cell->get_fe().dofs_per_cell; ++i)
          system_rhs(local_dof_indices[i]) += local_rhs(i);

        for(unsigned int i=0; i<cell->get_fe().dofs_per_cell;++i)
          for(unsigned int j=0; j<cell->get_fe().dofs_per_cell;++j)
            system_matrix.add (local_dof_indices[i],local_dof_indices[j],
                               local_matrix(i,j));
        std::map<types::global_dof_index,double> boundary_values;
        VectorTools::interpolate_boundary_values (dof_handler,
                                                  1,
                                                  StokesBoundaryValues<dim>(),
                                                  boundary_values,
                                                  fe_collection.component_mask(stokes_velocities));
        MatrixTools::apply_boundary_values (boundary_values,
                                            system_matrix, solution, system_rhs);

        if (cell_is_in_solid_domain (cell))
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->at_boundary(f) == false)
              {

                if ((cell->neighbor(f)->level() == cell->level())
                    &&
                    (cell->neighbor(f)->has_children() == false)
                    &&
                    cell_is_in_fluid_domain (cell->neighbor(f)))
                  {
                    darcy_fe_face_values.reinit (cell, f);

                    assemble_solid_interface_term (darcy_fe_face_values,
                                                   lm_phi, darcy_phi_u,
                                                   local_solid_interface_matrix);

                    for(unsigned int i=0; i<cell->get_fe().dofs_per_cell;++i)
                      for(unsigned int j=0; j<cell->get_fe().dofs_per_cell;++j)
                        system_matrix.add(local_dof_indices[i],local_dof_indices[j], local_solid_interface_matrix(i,j));
                  }

                else if ((cell->neighbor(f)->level() == cell->level())
                         &&
                         (cell->neighbor(f)->has_children() == true))
                  {
                    std::cout << "Hi!" << std::endl;
                    for (unsigned int subface=0;
                         subface<cell->face(f)->n_children();
                         ++subface)
                      if (cell_is_in_fluid_domain (cell->neighbor_child_on_subface
                                                   (f, subface)))
                        {
                          darcy_fe_subface_values.reinit (cell, f, subface);


                          assemble_solid_interface_term (darcy_fe_face_values,
                                                         lm_phi, darcy_phi_u,
                                                         local_solid_interface_matrix);
                          for(unsigned int i=0; i<cell->get_fe().dofs_per_cell;++i)
                            for(unsigned int j=0; j<cell->get_fe().dofs_per_cell;++j)
                              system_matrix.add(local_dof_indices[i],local_dof_indices[j], local_solid_interface_matrix(i,j));
                        }
                  }

                else if (cell->neighbor_is_coarser(f)
                         &&
                         cell_is_in_fluid_domain(cell->neighbor(f)))
                  {
                    std::cout << "Hi2!" << std::endl;
                    darcy_fe_face_values.reinit (cell, f);

                    assemble_solid_interface_term (darcy_fe_face_values,
                                                   lm_phi, darcy_phi_u,
                                                   local_solid_interface_matrix);

                    for(unsigned int i=0; i<cell->get_fe().dofs_per_cell;++i)
                      for(unsigned int j=0; j<cell->get_fe().dofs_per_cell;++j)
                        system_matrix.add(local_dof_indices[i],local_dof_indices[j], local_solid_interface_matrix(i,j));
                  }
              }



        if (cell_is_in_fluid_domain (cell))
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->at_boundary(f) == false)
              {

                if ((cell->neighbor(f)->level() == cell->level())
                    &&
                    (cell->neighbor(f)->has_children() == false)
                    &&
                    cell_is_in_solid_domain (cell->neighbor(f)))
                  {
                    stokes_fe_face_values.reinit (cell, f);
                    darcy_fe_face_values.reinit (cell->neighbor(f),
                                                 cell->neighbor_of_neighbor(f));

                    assemble_fluid_interface_term (darcy_fe_face_values, stokes_fe_face_values,
                                                   lm_phi, stokes_phi_u,
                                                   local_fluid_interface_matrix);

                    assemble_fluid_BJS_interface_term (stokes_fe_face_values, stokes_phi_u,
                                                       local_fluid_BJS_interface_matrix);

                    cell->neighbor(f)->get_dof_indices (neighbor_solid_dof_indices);

                    for(unsigned int i=0; i<cell->get_fe().dofs_per_cell;++i)
                      for(unsigned int j=0; j<cell->get_fe().dofs_per_cell;++j)
                        system_matrix.add(local_dof_indices[i],local_dof_indices[j], local_fluid_BJS_interface_matrix(i,j));

                    for(unsigned int i=0; i<cell->get_fe().dofs_per_cell;++i)
                      for(unsigned int j=0; j<cell->neighbor(f)->get_fe().dofs_per_cell;++j)
                        system_matrix.add(local_dof_indices[i],neighbor_solid_dof_indices[j], local_fluid_interface_matrix(i,j));

                    for(unsigned int i=0; i<cell->neighbor(f)->get_fe().dofs_per_cell;++i)
                      for(unsigned int j=0; j<cell->get_fe().dofs_per_cell;++j)
                        system_matrix.add(neighbor_solid_dof_indices[i],local_dof_indices[j], local_fluid_interface_matrix(j,i));

                  }


                else if ((cell->neighbor(f)->level() == cell->level())
                         &&
                         (cell->neighbor(f)->has_children() == true))
                  {
                    for (unsigned int subface=0;
                         subface<cell->face(f)->n_children();
                         ++subface)
                      if (cell_is_in_solid_domain (cell->neighbor_child_on_subface
                                                   (f, subface)))
                        {
                          stokes_fe_subface_values.reinit (cell,
                                                           f,
                                                           subface);
                          darcy_fe_face_values.reinit (cell->neighbor_child_on_subface (f, subface),
                                                       cell->neighbor_of_neighbor(f));

                          assemble_fluid_interface_term (darcy_fe_face_values, stokes_fe_face_values,
                                                         lm_phi, stokes_phi_u,
                                                         local_fluid_interface_matrix);

                          assemble_fluid_BJS_interface_term (stokes_fe_face_values, stokes_phi_u,
                                                             local_fluid_BJS_interface_matrix);

                          cell->neighbor_child_on_subface (f, subface)
                              ->get_dof_indices (neighbor_solid_dof_indices);

                          for(unsigned int i=0; i<cell->get_fe().dofs_per_cell;++i)
                            for(unsigned int j=0; j<cell->get_fe().dofs_per_cell;++j)
                              system_matrix.add(local_dof_indices[i],local_dof_indices[j], local_fluid_BJS_interface_matrix(i,j));

                          for(unsigned int i=0; i<cell->neighbor(f)->get_fe().dofs_per_cell;++i)
                            for(unsigned int j=0; j<cell->get_fe().dofs_per_cell;++j)
                              system_matrix.add(local_dof_indices[i],neighbor_solid_dof_indices[j], local_fluid_interface_matrix(i,j));

                        }
                  }

                else if (cell->neighbor_is_coarser(f)
                         &&
                         cell_is_in_solid_domain(cell->neighbor(f)))
                  {
                    stokes_fe_face_values.reinit (cell, f);
                    darcy_fe_subface_values.reinit (cell->neighbor(f),
                                                    cell->neighbor_of_coarser_neighbor(f).first,
                                                    cell->neighbor_of_coarser_neighbor(f).second);

                    assemble_fluid_interface_term (darcy_fe_face_values, stokes_fe_face_values,
                                                   lm_phi, stokes_phi_u,
                                                   local_fluid_interface_matrix);

                    assemble_fluid_BJS_interface_term (stokes_fe_face_values, stokes_phi_u,
                                                       local_fluid_BJS_interface_matrix);

                    cell->neighbor(f)->get_dof_indices (neighbor_solid_dof_indices);

                    for(unsigned int i=0; i<cell->get_fe().dofs_per_cell;++i)
                      for(unsigned int j=0; j<cell->get_fe().dofs_per_cell;++j)
                        system_matrix.add(local_dof_indices[i],local_dof_indices[j], local_fluid_BJS_interface_matrix(i,j));

                    for(unsigned int i=0; i<cell->neighbor(f)->get_fe().dofs_per_cell;++i)
                      for(unsigned int j=0; j<cell->get_fe().dofs_per_cell;++j)
                        system_matrix.add(local_dof_indices[i],neighbor_solid_dof_indices[j], local_fluid_interface_matrix(i,j));

                  }
              }
      }

  }



  template <int dim>
  void
  FluidStructureProblem<dim>::
  assemble_solid_interface_term (const FEFaceValuesBase<dim>          &darcy_fe_face_values,
                                 std::vector<Tensor<1,dim> >          &lm_phi,
                                 std::vector<Tensor<1,dim> >          &darcy_phi_u,
                                 FullMatrix<double>                   &local_solid_interface_matrix) const
  {
    //  Assert (darcy_fe_face_values.n_quadrature_points ==
    //          lm_fe_face_values.n_quadrature_points,
    //          ExcInternalError());
    const unsigned int n_face_quadrature_points
        = darcy_fe_face_values.n_quadrature_points;

    const FEValuesExtractors::Vector darcy_velocities (dim+1);
    const FEValuesExtractors::Vector l_multiplier (3*dim);

    local_solid_interface_matrix = 0;
    for (unsigned int q=0; q<n_face_quadrature_points; ++q)
      {
        const Tensor<1,dim> normal_vector = darcy_fe_face_values.normal_vector(q);

        for (unsigned int k=0; k<darcy_fe_face_values.dofs_per_cell; ++k)
          {
            darcy_phi_u[k] = darcy_fe_face_values[darcy_velocities].value (k, q);
            lm_phi[k] = darcy_fe_face_values[l_multiplier].value (k,q);
          }

        for (unsigned int i=0; i<darcy_fe_face_values.dofs_per_cell; ++i)
          for (unsigned int j=0; j<darcy_fe_face_values.dofs_per_cell; ++j)
            local_solid_interface_matrix(i,j) += ((darcy_phi_u[i] *normal_vector) *
                                                  (lm_phi[j] *normal_vector)
                                                  +(darcy_phi_u[j] *normal_vector) *(lm_phi[i] *normal_vector))*
                darcy_fe_face_values.JxW(q);
      }
  }

  template <int dim>
  void
  FluidStructureProblem<dim>::
  assemble_fluid_interface_term (const FEFaceValuesBase<dim>    &darcy_fe_face_values,
                                 const FEFaceValuesBase<dim>          &stokes_fe_face_values,
                                 std::vector<Tensor<1,dim> >          &lm_phi,
                                 std::vector<Tensor<1,dim> >          &stokes_phi_u,
                                 FullMatrix<double>                   &local_fluid_interface_matrix) const
  {
    Assert (stokes_fe_face_values.n_quadrature_points ==
            darcy_fe_face_values.n_quadrature_points,
            ExcInternalError());
    const unsigned int n_face_quadrature_points
        = darcy_fe_face_values.n_quadrature_points;

    const FEValuesExtractors::Vector stokes_velocities (0);
    const FEValuesExtractors::Vector l_multiplier (3*dim);

    local_fluid_interface_matrix = 0;
    for (unsigned int q=0; q<n_face_quadrature_points; ++q)
      {
        const Tensor<1,dim> normal_vector = stokes_fe_face_values.normal_vector(q);

        for (unsigned int k=0; k<stokes_fe_face_values.dofs_per_cell; ++k)
          stokes_phi_u[k] = stokes_fe_face_values[stokes_velocities].value (k, q);
        for (unsigned int k=0; k<darcy_fe_face_values.dofs_per_cell; ++k)
          lm_phi[k] = darcy_fe_face_values[l_multiplier].value (k,q);

        for (unsigned int i=0; i<stokes_fe_face_values.dofs_per_cell; ++i)
          for (unsigned int j=0; j<darcy_fe_face_values.dofs_per_cell; ++j)
            local_fluid_interface_matrix(i,j) += -(stokes_phi_u[i] *normal_vector) *
                (lm_phi[j] *normal_vector) *
                stokes_fe_face_values.JxW(q);
      }
  }

  template <int dim>
  void
  FluidStructureProblem<dim>::
  assemble_fluid_BJS_interface_term (const FEFaceValuesBase<dim>          &stokes_fe_face_values,
                                     std::vector<Tensor<1,dim> >          &stokes_phi_u,
                                     FullMatrix<double>                   &local_fluid_BJS_interface_matrix) const
  {
    const unsigned int n_face_quadrature_points
        = stokes_fe_face_values.n_quadrature_points;

    const FEValuesExtractors::Vector stokes_velocities (0);

    local_fluid_BJS_interface_matrix = 0;
    for (unsigned int q=0; q<n_face_quadrature_points; ++q)
      {
        const Tensor<1,dim> normal_vector = stokes_fe_face_values.normal_vector(q);

        for (unsigned int k=0; k<stokes_fe_face_values.dofs_per_cell; ++k)
          stokes_phi_u[k] = stokes_fe_face_values[stokes_velocities].value (k, q);

        for (unsigned int i=0; i<stokes_fe_face_values.dofs_per_cell; ++i)
          for (unsigned int j=0; j<stokes_fe_face_values.dofs_per_cell; ++j)
            local_fluid_BJS_interface_matrix(i,j) += alpha_bjs*viscosity*((stokes_phi_u[i] *stokes_phi_u[j])
                                                                          -(stokes_phi_u[i] *normal_vector)*(stokes_phi_u[j] *normal_vector))*
                stokes_fe_face_values.JxW(q);
      }
  }


  template <int dim>
  void
  FluidStructureProblem<dim>::solve ()
  {

    SparseDirectUMFPACK direct_solver;
    direct_solver.initialize (system_matrix);
    direct_solver.vmult (solution, system_rhs);

    std::cout << "Solution norm: " << solution.norm_sqr()<<std::endl;
  }



  template <int dim>
  void
  FluidStructureProblem<dim>::
  output_results ()  const
  {
    std::vector<std::string> solution_names (dim, "stokes_velocity");
    solution_names.push_back ("stokes_pressure");
    for (unsigned int d=0; d<dim; ++d)
      solution_names.push_back ("darcy_velocity");
    solution_names.push_back ("darcy_pressure");
    for (unsigned int d=0; d<dim; ++d)
      solution_names.push_back ("l_multiplier");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation
        (dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation
        .push_back (DataComponentInterpretation::component_is_scalar);
    for (unsigned int d=0; d<dim; ++d)
      data_component_interpretation
          .push_back (DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation
        .push_back (DataComponentInterpretation::component_is_scalar);
    for (unsigned int d=0; d<dim; ++d)
      data_component_interpretation
          .push_back (DataComponentInterpretation::component_is_part_of_vector);

    DataOut<dim,hp::DoFHandler<dim> > data_out;
    data_out.attach_dof_handler (dof_handler);

    data_out.add_data_vector (solution, solution_names,
                              DataOut<dim,hp::DoFHandler<dim> >::type_dof_data,
                              data_component_interpretation);
    data_out.build_patches ();

    std::ostringstream filename;
    filename << "solution"
             << ".vtk";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtk (output);
  }


  template <int dim>
  void FluidStructureProblem<dim>::run ()
  {
    make_grid ();

    setup_dofs ();

    std::cout << "   Assembling..." << std::endl;
    assemble_system ();


    std::cout << "   Solving..." << std::endl;
    solve ();

    std::cout << "   Writing output..." << std::endl;
    output_results ();

    std::cout << std::endl;

  }
}



int main ()
{
  try
  {
    using namespace dealii;
    using namespace StokesDarcy;

    FluidStructureProblem<2> flow_problem(1, 1);
    flow_problem.run ();
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
