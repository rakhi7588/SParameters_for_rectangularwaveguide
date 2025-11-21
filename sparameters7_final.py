import pyvista as pv
import numpy as np

class WaveguideSParameterCalculator:
    
    # Global boundary variables for ports
    INPUT_PORT_Z_BOUNDARY = -0.012
    OUTPUT_PORT_X_BOUNDARY = 0.016
    TOLERANCE = 5e-9
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.mesh = None
        self.surface = None
        self.input_port = None
        self.output_port = None
    
    #Mesh is read.
    def read_mesh(self):
        self.mesh = pv.read(self.file_path)#Loads the 3D simulation result file (a mesh with field data)
        print("Before surface extraction:")
        print("Mesh points shape:", self.mesh.points.shape)#Prints how many points are in the mesh before further processing.
        
    # Extract only the outer surface mesh from the volume mesh
    def extract_surface(self):
        #extract_surface() takes a volumetric mesh and returns a surface PolyData mesh that represents the shape’s external boundary, 
        #PolyData meshes are 2D manifolds used for visualization and surface operations. It contains:
        #Points: The vertices on the outer surface.
        # Polygons (cells): Surface faces connecting those points.
        #Point data: Scalars or vectors at points (e.g., coordinates, electric/magnetic field values).
        #Cell data: Scalars or vectors defined for each polygon cell.
        self.surface = self.mesh.extract_surface()#Converts the full 3D mesh into just the outer surface (like peeling the skin off the geometry
        print("\nAfter surface extraction:")
        print("Surface points shape:", self.surface.points.shape)
        
        self.surface['x_coord'] = self.surface.points[:, 0]#selects all the x-coordinates of every point.
        self.surface['z_coord'] = self.surface.points[:, 2]#selects all the z-coordinates of every point.
    
    # Extract input and output port surfaces by thresholding coordinates
    def extract_ports(self):
        self.input_port = self.surface.threshold(
            [self.INPUT_PORT_Z_BOUNDARY - self.TOLERANCE, self.INPUT_PORT_Z_BOUNDARY + self.TOLERANCE], 
            scalars='z_coord', all_scalars=True)#Cuts out just the parts of the surface that lie at the input  boundaries (based on x or z position).
        self.output_port = self.surface.threshold(
            [self.OUTPUT_PORT_X_BOUNDARY - self.TOLERANCE, self.OUTPUT_PORT_X_BOUNDARY + self.TOLERANCE], 
            scalars='x_coord', all_scalars=True)#Cuts out just the parts of the surface that lie at the  output boundaries (based on x or z position)
        
        # Extract surface meshes of input/output ports
        #Even though input_port and output_port are obtained by thresholding the full surface mesh, 
        # they might still be of mixed cell types or contain internal faces because thresholding returns a subset of the mesh.
        #Calling .extract_surface() on these thresholded subsets cleans them by:Extracting only the outer visible 2D boundary surface of these 
        # sub-meshes and removing any hidden or internal faces that may remain after thresholding
        #The result is a clean surface mesh of type PolyData, containing: Vertices (points) on the boundary.Filtered polygonal surface cells
        # (triangles, quads) covering the port geometry.Associated scalar and vector data on points and cells.
        self.input_port = self.input_port.extract_surface()
        self.output_port = self.output_port.extract_surface()
        
        print("\nNumber of input port points:", self.input_port.n_points)#These represent the number of points (nodes) in the mesh of the input port 
        print("Number of output port points:", self.output_port.n_points)#These represent the number of points (nodes) in the mesh of the output port 
        print("Available point fields on input port:", self.input_port.point_data.keys())#Returns the keys (names) of all scalar or vector data arrays associated
        #with each point on the input port mesh. Example Examples: electric field vectors, Poynting vector components
        print("Available cell fields on input port:", self.input_port.cell_data.keys())#Returns the keys of all scalar or vector data arrays associated with each 
        #cell (polygon) on the input port mesh.The "e" is simply a label indicating these are electromagnetic quantities on cells.
    
    # Function to integrate power using cell-based Poynting vector data and exact cell areas
    def integrate_port_power_cell_data(self, port, field_name):
        # function returns total power perpendicular to the port face
        # Compute cell areas for weighting integration
        port = port.compute_cell_sizes(length=False, area=True, volume=False)#compute_cell_sizes calculates the area of every surface cell on your port mesh for integration.
        areas = port.cell_data['Area']#It adds a new array called 'Area' to the cell data in the mesh, giving the precise physical area of each cell.

        # Extract cell vector field and compute cell normals
        #This retrieves a vector field from the cell data of the port mesh.
        S_cells = port.cell_data[field_name]#Here, field_name is typically "poynting vector re e", the real part of the Poynting vector stored for each mesh cell.
        #S_cells is an array containing the 3D Poynting vector values (power flow density vectors) defined on each polygonal cell of the port surface.
        cell_normals = port.compute_normals(cell_normals=True, point_normals=False).cell_normals#This computes the normal vectors to each cell of the port surface mesh
        #cell_normals=True means calculate one normal vector per cell (polygonal face).
        # point_normals=False disables normal calculation at points.
        #The function returns a new mesh with normals computed; .cell_normals accesses the computed normals array.
        #cell_normals is an array where each entry is the outward-pointing normal vector of a cell face.
        
        
        total_power = 0.0
        # Loop over all cells for numerical power integration
        #port.n_cells: This is the total number of cells (polygons) in the mesh for the port surface. 
        # A "cell" here is a polygonal face of the mesh, typically a triangle or quad.
        for cell_id in range(port.n_cells):#This loop iterates over every cell in the port mesh, indexed from 0 up to port.n_cells - 1. 
                                           #Each iteration processes one polygonal cell.
            poy_vec = S_cells[cell_id]#cell_id is just the number/id of the current polygon being processed on the mesh.
        #S_cells is an array of 3D vectors representing the Poynting vector (electromagnetic power flow density) on each cell.
        #poy_vec tells you the direction and magnitude of electromagnetic power flowing in that cell.
        #normal tells you which way the polygon of that cell is facing (pointing outward).
            normal = cell_normals[cell_id]#cell_normals is an array containing the normal vectors (perpendicular vectors) for each cell, computed earlier.
            normal_component = np.abs(np.dot(poy_vec, normal))#This calculates the dot product between two vectors: the Poynting vector S representing electromagnetic power flow at the cell, and the normal vector 
            # n of the cell's surface.
            area = areas[cell_id] #This gets the surface area of the current cell. Calculating the cell's physical area is crucial because larger cells carry proportionally more power.
            power_contrib = normal_component * area#The power flowing through this cell equals the perpendicular power density times the cell surface area — the local power contribution.
            total_power += power_contrib#This adds the cell's contribution to the total power passing through the entire port surface by summing over all polygonal cells.
        
        print("Total power integrated:", total_power)
        return total_power
    
    def calculate_s_parameters(self):
        # Integrate power on input and output port surfaces
        power_input = self.integrate_port_power_cell_data(self.input_port, "poynting vector re e")
        power_output = self.integrate_port_power_cell_data(self.output_port, "poynting vector re e")
        
        # Calculate incident, transmitted, and reflected powers
        incident_power = np.abs(power_input)
        transmitted_power = np.abs(power_output)
        reflected_power = incident_power - transmitted_power
        reflected_power = max(reflected_power, 0)
        
        print(f"\nTotal input power: {power_input} W")
        print(f"Total output power: {power_output} W")
        print(f"Incident power at input port: {incident_power:.6e} W")
        print(f"Transmitted power at output port: {transmitted_power:.6e} W")
        print(f"Reflected power at input port: {reflected_power:.6e} W")
        
        # Calculate S-parameters and convert to dB scale
        S21 = np.sqrt(transmitted_power / incident_power)
        S11 = np.sqrt(reflected_power / incident_power)
        
        S21_db = 20 * np.log10(S21)
        S11_db = 20 * np.log10(S11 + 1e-20)# Avoid log(0)
        
        print(f"\n|S21| (Transmission magnitude): {S21:.6f} ({S21_db:.2f} dB)")
        print(f"|S11| (Reflection magnitude): {S11:.6f} ({S11_db:.2f} dB)")
        
        loss_db = 10 * np.log10(incident_power / transmitted_power)
        print(f"Power loss through the waveguide: {loss_db:.2f} dB")

        return {
            "input_power": power_input,
            "output_power": power_output,
            "incident_power": incident_power,
            "transmitted_power": transmitted_power,
            "reflected_power": reflected_power,
            "S21": S21,
            "S11": S11,
            "S21_db": S21_db,
            "S11_db": S11_db,
            "loss_db": loss_db
        }
    
    # Visualization of ports and Poynting vector magnitudes for explanation
    def visualize_ports(self):
        plotter = pv.Plotter(shape=(1, 2), window_size=[1200, 600])
        
        print(self.input_port.cell_data["poynting vector re e"])
        print("output\n")
        print(self.output_port.cell_data["poynting vector re e"])
        plotter.subplot(0, 0)
        plotter.add_text("Input Port Surface with Poynting Vector Magnitude", font_size=12)
        mag_poy = np.linalg.norm(self.input_port.cell_data["poynting vector re e"], axis=1)
        self.input_port.cell_data['Poynting Mag'] = mag_poy
        plotter.add_mesh(self.input_port, scalars='Poynting Mag', cmap='viridis', show_scalar_bar=True)
        plotter.add_mesh(self.input_port.outline(), color='k')
        
        
        plotter.subplot(0, 1)
        plotter.add_text("Output Port Surface with Poynting Vector Magnitude", font_size=12)
        mag_poy_out = np.linalg.norm(self.output_port.cell_data["poynting vector re e"], axis=1)
        self.output_port.cell_data['Poynting Mag'] = mag_poy_out
        #plotter.add_mesh(self.input_port, scalars='Poynting Mag', cmap='viridis', show_scalar_bar=True)
        #plotter.add_mesh(self.input_port.outline(), color='k')
        
        print(mag_poy)
        print("\n")
        print(mag_poy_out)
        
        plotter.add_mesh(self.output_port, scalars='Poynting Mag', cmap='viridis', show_scalar_bar=True)
        plotter.add_mesh(self.output_port.outline(), color='b')
        
        plotter.link_views()
        plotter.show()

#Runs all the steps: read → extract → integrate → calculate → visualize.
def main():
    file_path = "/Users/rakhijha/Desktop/bendedRW/case_t0001.vtu"  #This is just the location of your .vtu simulation file.
    #A calculator object is created.
    #Calls the class’s __init__ method.
    #Stores the file path inside the object.
    #Initializes variables (mesh, surface, input_port, output_port) to None.
    calculator = WaveguideSParameterCalculator(file_path)
    calculator.read_mesh()#Uses PyVista (pv.read) to open the .vtu file. The file is parsed, and all the geometry + field data are now in 
    #memory inside calculator.mesh. It prints the number of points in the full 3D mesh.
    calculator.extract_surface()#Strips away the volume, keeping only the outer surface mesh.
                                #Adds helper arrays x_coord and z_coord to each point for easy filtering.
    calculator.extract_ports()#Cuts the surface to find the “input plane” and “output plane” of your waveguide. These are stored as separate 
    #meshes (input_port, output_port).Prints how many points/cells and what fields exist on those slices.
    results = calculator.calculate_s_parameters()#Integrates the Poynting vector over each port surface (this gives power)
    #From that, calculates |S11| (reflection) and |S21| (transmission) in both linear scale and dB. Prints results.
    calculator.visualize_ports()
    
#In Python, every file (module) has a special variable called __name__.When we run the file directly, Python sets __name__ to "__main__"
if __name__ == "__main__": #is used so that the script runs the main() function only when you execute the file directly, not when it's imported into another script as a module.
    main()#That condition is True, so Python calls main()

#File is read → main() runs → calculator is created → mesh file is loaded → surface is extracted → ports are cut → power is integrated → S-parameters are computed → plots are shown.