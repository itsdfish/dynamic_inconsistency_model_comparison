# dynamic_inconsistency_model_comparison

Model analysis scripts comparing three models of dynamic inconsistency: reference point priority heuristic, reference point prospect theory, and a quantum cognition model.

# Installation

1. [Download](https://julialang.org/downloads/) Julia 1.11 or higher.
2. Optionally install [VSCode](https://code.visualstudio.com/docs/languages/julia)
3. Run `git clone https://github.com/itsdfish/dynamic_inconsistency_model_comparison` to download repository 
4. Open Julia and run the following code to install the registry:

```julia 
using Pkg
pkg"registry add https://github.com/itsdfish/Registry.jl"
```
5. To install dependencies, cd to the project directory and in Julia activate the project environment by typing `]` to enter package mode. Enter `activate .` and enter `update` to install dependencies. 

Once the dependencies have been installed, you can run the scripts. 

Distribution A: Approved for public release. Case number: AFRL-2025-2842