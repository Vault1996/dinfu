#pragma once


#include "OptSolver.h"
#include "CombinedSolverParameters.h"
#include "SolverIteration.h"
#include "Config.h"

class CombinedSolverBase {
public:
    virtual void combinedSolveInit() = 0;

    virtual void combinedSolveFinalize() = 0;

    virtual void preSingleSolve() = 0;

    virtual void postSingleSolve() = 0;

    virtual void preNonlinearSolve(int iteration) = 0;

    virtual void postNonlinearSolve(int iteration) = 0;

    virtual void solveAll() {
        combinedSolveInit();
        for (auto s : m_solverInfo) {
            if (s.enabled) {
                preSingleSolve();
                if (m_combinedSolverParameters.numIter == 1) {
                    preNonlinearSolve(0);
                    s.solver->solve(m_solverParams, m_problemParams, m_combinedSolverParameters.profileSolve,
                                    s.iterationInfo);
                    postNonlinearSolve(0);
                } else {
                    for (int i = 0; i < (int) m_combinedSolverParameters.numIter; ++i) {
                        preNonlinearSolve(i);
                        s.solver->solve(m_solverParams, m_problemParams, m_combinedSolverParameters.profileSolve,
                                        s.iterationInfo);
                        postNonlinearSolve(i);
                        if (m_combinedSolverParameters.earlyOut) {
                            break;
                        }
                    }
                }
                postSingleSolve();
            }
        }
        combinedSolveFinalize();
    }

    double getCost(const std::string &name) {
        for (auto s : m_solverInfo) {
            if (s.name == name) {
                if (s.solver && s.enabled) {
                    return s.solver->finalCost();
                }
            }
        }
        return nan("");
    }

    void addSolver(std::shared_ptr<SolverBase> solver, std::string name, bool enabled = true) {
        m_solverInfo.resize(m_solverInfo.size() + 1);
        m_solverInfo[m_solverInfo.size() - 1].set(solver, std::move(name), enabled);

    }

    void addOptSolvers(std::vector<unsigned int> dims, std::string problemFilename, bool doublePrecision = false) {
        if (m_combinedSolverParameters.useOpt) {
            addSolver(std::make_shared<OptSolver>(dims, problemFilename, "gaussNewtonGPU", doublePrecision), "Opt(GN)",
                      true);
        }
        if (m_combinedSolverParameters.useOptLM) {
            addSolver(std::make_shared<OptSolver>(dims, problemFilename, "LMGPU", doublePrecision), "Opt(LM)", true);
        }
    }


protected:
    struct SolverInfo {
        std::shared_ptr<SolverBase> solver;
        std::vector<SolverIteration> iterationInfo;
        std::string name;
        bool enabled;

        void set(std::shared_ptr<SolverBase> _solver, std::string _name, bool _enabled) {
            solver = std::move(_solver);
            name = std::move(_name);
            enabled = _enabled;
        }
    };

    std::vector<SolverInfo> m_solverInfo;

    NamedParameters m_solverParams;
    NamedParameters m_problemParams;
    CombinedSolverParameters m_combinedSolverParameters;
};