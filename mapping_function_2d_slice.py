# Add this after your existing 3D plot code (after plt.show() for 3D)

# Filter for 2D slice at y ≈ 0.5 (e.g., between 0.45 and 0.55)
slice_data = sampled[(sampled['y'] >= 0.45) & (sampled['y'] <= 0.55)]

# Plot 2D slice
fig2d = plt.figure(figsize=(8, 6))
ax2d = fig2d.add_subplot(111)
ax2d.scatter(slice_data['x'], slice_data['z'], s=10, alpha=0.7)
ax2d.set_xlabel('Total Inventory Usage Rate (utilization)')
ax2d.set_ylabel('EFP Usage Rate (z)')
ax2d.set_title('2D Slice at y ≈ 0.5 (EFP AQ / Total AQ ≈ 0.5)')
ax2d.grid(True)
plt.tight_layout()
plt.savefig(r'D:\project_part2\mapping_function_2d_slice_y0.5.png', dpi=300)  # Optional: Save the 2D plot
plt.show()