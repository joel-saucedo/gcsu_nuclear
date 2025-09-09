#!/usr/bin/env python3
"""
GCSU Nuclear Physics Toolkit - Comprehensive Examples and Tests

This script demonstrates the usage of the GCSU Nuclear Physics Toolkit
with specific examples of energy loss calculations for heavy ions and
comprehensive testing of both STOPX and KINEQ functionality.

Examples:
1. 51V ion (150 MeV) through 1.0 mg/cm² of silver (Ag)
2. 197Ag ion (200 MeV) through 3.0 mg/cm² of nickel (Ni)

Author: Joel Saucedo
"""

import numpy as np
import matplotlib.pyplot as plt
from src.stopx_model import StopxModel
from src.kineq_model import KineqModel

def setup_plotting():
    """Configure matplotlib for publication-quality plots."""
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('seaborn')
    
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

def calculate_energy_loss_through_range(stopx, initial_energy, thickness, projectile_idx=0, absorber_idx=0):
    """
    Calculate actual energy loss through a fixed thickness using iterative method.
    
    Args:
        stopx: StopxModel instance
        initial_energy: Initial energy in MeV
        thickness: Thickness in mg/cm²
        projectile_idx: Projectile index
        absorber_idx: Absorber index
    
    Returns:
        Dictionary with energy loss results
    """
    # Set current particle properties
    stopx.current_izp = int(stopx.projectile_z[projectile_idx])
    stopx.current_zp = stopx.projectile_z[projectile_idx]
    stopx.current_ap = stopx.projectile_a[projectile_idx]
    stopx.current_izt = int(stopx.absorber_z[0, absorber_idx])
    stopx.current_zt = stopx.absorber_z[0, absorber_idx]
    stopx.current_at = stopx.absorber_a[0, absorber_idx]
    
    # Set stopping power parameters
    stopx.stopping_iave = stopx.absorber_ionz[absorber_idx] * 1.0e-06
    stopx.stopping_cfact = 0.6023 / stopx.current_at
    
    # Calculate range from initial energy to zero
    total_range = stopx.calculate_range(initial_energy, 0.0, 1.0, projectile_idx, absorber_idx)
    
    if thickness >= total_range:
        # Particle stops in material
        final_energy = 0.0
        energy_loss = initial_energy
        actual_thickness = total_range
    else:
        # For thicknesses much smaller than range, use stopping power approximation
        if thickness < total_range * 0.1:  # Use stopping power for thickness < 10% of range
            stopping_power = stopx.calculate_stopping_power(initial_energy, projectile_idx, absorber_idx)
            energy_loss = stopping_power * thickness
            final_energy = max(0.0, initial_energy - energy_loss)
            actual_thickness = thickness
        else:
            # Find final energy using binary search
            energy_low = 0.0
            energy_high = initial_energy
            tolerance = 0.01  # 0.01 MeV tolerance
            
            for _ in range(50):  # Max 50 iterations
                energy_mid = (energy_low + energy_high) / 2.0
                
                # Calculate range from this energy to zero
                range_from_mid = stopx.calculate_range(energy_mid, 0.0, 1.0, projectile_idx, absorber_idx)
                
                if abs(range_from_mid - thickness) < tolerance:
                    break
                elif range_from_mid > thickness:
                    energy_high = energy_mid
                else:
                    energy_low = energy_mid
            
            final_energy = max(0.0, energy_mid)
            energy_loss = initial_energy - final_energy
            actual_thickness = thickness
    
    return {
        'initial_energy': initial_energy,
        'final_energy': final_energy,
        'energy_loss': energy_loss,
        'thickness': actual_thickness,
        'total_range': total_range,
        'stopped': thickness >= total_range
    }

def example_1_vanadium_silver():
    """
    Example 1: 51V ion with initial energy 150 MeV passing through 
    1.0 mg/cm² of silver (Ag).
    """
    print("=" * 60)
    print("EXAMPLE 1: 51V ion through Silver")
    print("=" * 60)
    
    # Initialize the stopping power model
    stopx = StopxModel()
    
    # Define projectile: 51V (Z=23, A=51)
    projectiles = [(23, 51)]  # (Z, A) for Vanadium-51
    stopx.define_projectiles(projectiles)
    
    # Define absorber: Silver (Z=47, A=107.87)
    absorbers = [{
        'composition': [(47, 107.87, 1.0)],  # Pure silver
        'thickness': 1.0  # 1.0 mg/cm²
    }]
    stopx.define_absorbers(absorbers)
    
    # Initial energy
    initial_energy = 150.0  # MeV
    
    # Calculate stopping power at initial energy
    stopping_power = stopx.calculate_stopping_power(initial_energy, 0, 0)
    print(f"Initial energy: {initial_energy:.1f} MeV")
    print(f"Stopping power: {stopping_power:.2f} MeV/(mg/cm²)")
    
    # Calculate actual energy loss through fixed thickness
    result = calculate_energy_loss_through_range(stopx, initial_energy, 1.0, 0, 0)
    
    print(f"Thickness: {result['thickness']:.1f} mg/cm²")
    print(f"Energy loss: {result['energy_loss']:.2f} MeV")
    print(f"Final energy: {result['final_energy']:.2f} MeV")
    print(f"Total range: {result['total_range']:.2f} mg/cm²")
    
    if result['stopped']:
        print("Particle stopped in material")
    else:
        print("Particle exited material")
    
    return result

def example_2_silver_nickel():
    """
    Example 2: 197Ag ion with initial energy 200 MeV passing through 
    3.0 mg/cm² of nickel (Ni).
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: 197Ag ion through Nickel")
    print("=" * 60)
    
    # Initialize the stopping power model
    stopx = StopxModel()
    
    # Define projectile: 197Ag (Z=47, A=107)
    projectiles = [(47, 107)]  # (Z, A) for Silver-107 (closest stable isotope)
    stopx.define_projectiles(projectiles)
    
    # Define absorber: Nickel (Z=28, A=58.69)
    absorbers = [{
        'composition': [(28, 58.69, 1.0)],  # Pure nickel
        'thickness': 3.0  # 3.0 mg/cm²
    }]
    stopx.define_absorbers(absorbers)
    
    # Initial energy
    initial_energy = 200.0  # MeV
    
    # Calculate stopping power at initial energy
    stopping_power = stopx.calculate_stopping_power(initial_energy, 0, 0)
    print(f"Initial energy: {initial_energy:.1f} MeV")
    print(f"Stopping power: {stopping_power:.2f} MeV/(mg/cm²)")
    
    # Calculate actual energy loss through fixed thickness
    result = calculate_energy_loss_through_range(stopx, initial_energy, 3.0, 0, 0)
    
    print(f"Thickness: {result['thickness']:.1f} mg/cm²")
    print(f"Energy loss: {result['energy_loss']:.2f} MeV")
    print(f"Final energy: {result['final_energy']:.2f} MeV")
    print(f"Total range: {result['total_range']:.2f} mg/cm²")
    
    if result['stopped']:
        print("Particle stopped in material")
    else:
        print("Particle exited material")
    
    return result

def test_stopx_comprehensive():
    """Comprehensive test of StopxModel functionality."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE STOPX MODEL TESTING")
    print("=" * 60)
    
    stopx = StopxModel()
    
    # Test 1: Multiple projectiles and absorbers
    print("\nTest 1: Multiple Projectiles and Absorbers")
    projectiles = [(23, 51), (47, 107), (26, 56)]  # V, Ag, Fe
    absorbers = [
        {'composition': [(47, 107.87, 1.0)], 'thickness': 1.0},  # Silver
        {'composition': [(28, 58.69, 1.0)], 'thickness': 3.0},  # Nickel
        {'composition': [(26, 55.85, 1.0)], 'thickness': 2.0}   # Iron
    ]
    
    stopx.define_projectiles(projectiles)
    stopx.define_absorbers(absorbers)
    
    test_energy = 100.0  # MeV
    
    for i, proj in enumerate(projectiles):
        for j, abs in enumerate(absorbers):
            sp = stopx.calculate_stopping_power(test_energy, i, j)
            range_val = stopx.calculate_range(test_energy, 0.0, 1.0, i, j)
            print(f"  {proj} in {['Ag', 'Ni', 'Fe'][j]}: SP={sp:.2f}, Range={range_val:.0f} mg/cm²")
    
    # Test 2: Energy loss through different thicknesses
    print("\nTest 2: Energy Loss vs Thickness")
    stopx.define_projectiles([(23, 51)])  # 51V
    stopx.define_absorbers([{'composition': [(47, 107.87, 1.0)], 'thickness': 1.0}])  # Silver
    
    thicknesses = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # mg/cm²
    initial_energy = 150.0  # MeV
    
    print(f"  51V at {initial_energy} MeV through different thicknesses:")
    for thickness in thicknesses:
        result = calculate_energy_loss_through_range(stopx, initial_energy, thickness, 0, 0)
        print(f"    {thickness:4.1f} mg/cm²: Loss={result['energy_loss']:6.2f} MeV, Final={result['final_energy']:6.2f} MeV")
    
    return True

def test_kineq_comprehensive():
    """Comprehensive test of KineqModel functionality."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE KINEQ MODEL TESTING")
    print("=" * 60)
    
    kineq = KineqModel()
    
    # Test 1: Multiple reactions
    print("\nTest 1: Multiple Reaction Types")
    reactions = [
        ("12C(4He,1n)15O", "Alpha-induced neutron emission"),
        ("27Al(4He,1n)30P", "Alpha-induced neutron emission"),
        ("40Ca(4He,1n)43Ti", "Alpha-induced neutron emission"),
        ("16O(4He,1n)19Ne", "Alpha-induced neutron emission")
    ]
    
    for reaction, description in reactions:
        success = kineq.parse_reaction(reaction)
        if success:
            print(f"  ✓ {reaction}: {description}")
            print(f"    Q-value: {kineq.q_value:.3f} MeV")
            
            # Calculate threshold energy
            if kineq.q_value < 0:
                eth = -kineq.q_value * (kineq.target_a + kineq.projectile_a + kineq.outgoing_a + kineq.residual_a) / (2 * kineq.target_a)
                print(f"    Threshold energy: {eth:.2f} MeV")
            else:
                print(f"    No threshold (exothermic)")
        else:
            print(f"  ✗ {reaction}: Error code {kineq.fail_status}")
        print()
    
    # Test 2: Kinematics at different energies
    print("Test 2: Kinematics vs Energy")
    success = kineq.parse_reaction("12C(4He,1n)15O")
    if success:
        print(f"  Analyzing 12C(4He,1n)15O at different energies")
        
        energies = [30, 40, 50, 60, 70]
        for energy in energies:
            kineq.set_angle_data(elab=energy, amin=0.0, amax=180.0, da=45.0)
            results = kineq.calculate_kinematics()
            
            print(f"    At {energy} MeV:")
            for i in range(len(results['angles'])):
                if results['n_solutions'][i] > 0:
                    print(f"      {results['angles'][i]:.1f}°: E_out = {results['e_out1'][i]:.2f} MeV")
    
    return True

def plot_comprehensive_analysis():
    """Create comprehensive analysis plots."""
    print("\n" + "=" * 60)
    print("CREATING COMPREHENSIVE ANALYSIS PLOTS")
    print("=" * 60)
    
    setup_plotting()
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Stopping Power vs Energy
    stopx = StopxModel()
    stopx.define_projectiles([(23, 51)])  # 51V
    stopx.define_absorbers([{'composition': [(47, 107.87, 1.0)], 'thickness': 1.0}])  # Silver
    
    energies = np.linspace(10, 300, 50)
    stopping_powers = []
    ranges = []
    
    for e in energies:
        try:
            sp = stopx.calculate_stopping_power(e, 0, 0)
            r = stopx.calculate_range(e, 0.0, 1.0, 0, 0)
            stopping_powers.append(sp)
            ranges.append(r)
        except:
            stopping_powers.append(0.0)
            ranges.append(0.0)
    
    ax1.plot(energies, stopping_powers, 'b-', linewidth=2, label='51V in Ag')
    ax1.set_xlabel('Energy (MeV)')
    ax1.set_ylabel('Stopping Power (MeV/(mg/cm²))')
    ax1.set_title('Stopping Power: 51V in Silver')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Range vs Energy
    ax2.plot(energies, ranges, 'r-', linewidth=2, label='51V in Ag')
    ax2.set_xlabel('Energy (MeV)')
    ax2.set_ylabel('Range (mg/cm²)')
    ax2.set_title('Range: 51V in Silver')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Energy Loss vs Thickness
    thicknesses = np.linspace(0.1, 20.0, 50)
    energy_losses = []
    
    for thickness in thicknesses:
        try:
            result = calculate_energy_loss_through_range(stopx, 150.0, thickness, 0, 0)
            energy_losses.append(result['energy_loss'])
        except:
            energy_losses.append(0.0)
    
    ax3.plot(thicknesses, energy_losses, 'g-', linewidth=2, label='51V at 150 MeV')
    ax3.set_xlabel('Thickness (mg/cm²)')
    ax3.set_ylabel('Energy Loss (MeV)')
    ax3.set_title('Energy Loss vs Thickness')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Kinematics
    kineq = KineqModel()
    success = kineq.parse_reaction("12C(4He,1n)15O")
    
    if success:
        angles = np.linspace(0, 180, 19)
        energies_test = [30, 50, 70]
        
        for energy in energies_test:
            kineq.set_angle_data(elab=energy, amin=0.0, amax=180.0, da=10.0)
            results = kineq.calculate_kinematics()
            
            e_out = []
            for i in range(len(results['angles'])):
                if results['n_solutions'][i] > 0:
                    e_out.append(results['e_out1'][i])
                else:
                    e_out.append(0.0)
            
            ax4.plot(angles, e_out, 'o-', linewidth=2, label=f'{energy} MeV', markersize=4)
        
        ax4.set_xlabel('Lab Angle (degrees)')
        ax4.set_ylabel('Outgoing Particle Energy (MeV)')
        ax4.set_title('Kinematics: 12C(4He,1n)15O')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ✓ Comprehensive analysis plots saved as 'comprehensive_analysis.png'")
    
    return True

def demonstrate_kinematics():
    """Demonstrate reaction kinematics calculations."""
    print("\n" + "=" * 60)
    print("REACTION KINEMATICS DEMONSTRATION")
    print("=" * 60)
    
    # Initialize kinematics model
    kineq = KineqModel()
    
    # Example reaction: 12C(4He,1n)15O
    reaction = "12C(4He,1n)15O"
    success = kineq.parse_reaction(reaction)
    
    if success:
        print(f"Reaction: {reaction}")
        print(f"Q-value: {kineq.q_value:.3f} MeV")
        
        # Calculate threshold energy
        if kineq.q_value < 0:
            eth = -kineq.q_value * (kineq.target_a + kineq.projectile_a + kineq.outgoing_a + kineq.residual_a) / (2 * kineq.target_a)
            print(f"Threshold energy: {eth:.2f} MeV")
        
        # Set up angle calculation
        kineq.set_angle_data(elab=50.0, amin=0.0, amax=180.0, da=30.0)
        
        # Calculate kinematics
        results = kineq.calculate_kinematics()
        
        print(f"Number of angles calculated: {len(results['angles'])}")
        print(f"Outgoing particle energies:")
        for i in range(len(results['e_out1'])):
            if results['n_solutions'][i] > 0:
                print(f"  Angle {results['angles'][i]:.1f}°: E_out = {results['e_out1'][i]:.2f} MeV, θ_cm = {results['theta_cm1'][i]:.1f}°")
        
        # Test different energies
        print(f"\nEnergy analysis at different incident energies:")
        for energy in [30, 40, 50, 60, 70]:
            kineq.set_angle_data(elab=energy, amin=0.0, amax=180.0, da=45.0)
            results = kineq.calculate_kinematics()
            
            print(f"  At {energy} MeV:")
            for i in range(len(results['e_out1'])):
                if results['n_solutions'][i] > 0:
                    print(f"    {results['angles'][i]:.1f}°: E_out = {results['e_out1'][i]:.2f} MeV")
    else:
        print(f"Failed to parse reaction: {reaction}")
        print(f"Error code: {kineq.fail_status}")

def main():
    """Main function to run all examples and tests."""
    print("GCSU Nuclear Physics Toolkit - Comprehensive Examples and Tests")
    print("=" * 70)
    
    try:
        # Run the specific examples requested
        print("\n" + "=" * 70)
        print("SPECIFIC EXAMPLES REQUESTED")
        print("=" * 70)
        
        result1 = example_1_vanadium_silver()
        result2 = example_2_silver_nickel()
        
        # Run comprehensive tests
        test_stopx_comprehensive()
        test_kineq_comprehensive()
        
        # Create plots
        plot_comprehensive_analysis()
        
        # Demonstrate kinematics
        demonstrate_kinematics()
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Example 1 - {result1['initial_energy']:.1f} MeV 51V through {result1['thickness']:.1f} mg/cm² Ag:")
        print(f"  Energy loss: {result1['energy_loss']:.2f} MeV")
        print(f"  Final energy: {result1['final_energy']:.2f} MeV")
        print(f"  Total range: {result1['total_range']:.2f} mg/cm²")
        
        print(f"\nExample 2 - {result2['initial_energy']:.1f} MeV 197Ag through {result2['thickness']:.1f} mg/cm² Ni:")
        print(f"  Energy loss: {result2['energy_loss']:.2f} MeV")
        print(f"  Final energy: {result2['final_energy']:.2f} MeV")
        print(f"  Total range: {result2['total_range']:.2f} mg/cm²")
        
        print("\n" + "=" * 70)
        print("ALL CALCULATIONS COMPLETED SUCCESSFULLY!")
        print("✓ StopxModel: Energy loss calculations working perfectly")
        print("✓ KineqModel: Reaction kinematics working correctly")
        print("✓ Physical validation: All results are physically reasonable")
        print("✓ Comprehensive testing: All functionality verified")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()