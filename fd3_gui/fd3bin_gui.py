import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from tkinter import filedialog
import subprocess

import pandas as pd
import numpy as np
from scipy import interpolate
import os
import fnmatch
from astropy.io import fits

import shutil
from datetime import datetime
import re
import time

class Application(tk.Tk):
    def __init__(self):
        
        tk.Tk.__init__(self)
        self.state('zoomed')
        self.title("PYfd3")


        left_frame = tk.Frame(self, width=self.winfo_screenwidth()//4)
        mid_frame = tk.Frame(self, width=self.winfo_screenwidth()//4)
        right_frame = tk.Frame(self)

        left_frame.pack(side='left', fill='both', expand=True)
        mid_frame.pack(side='left', fill='both', expand=True)
        right_frame.pack(side='left', fill='both', expand=True)



        self.change_path_button = tk.Button(left_frame, text='Change Working Dir', command=self.change_path)
        self.n_spec_label = tk.Label(left_frame, text='N Spec')
        self.text_ext = tk.Entry(left_frame)
        self.text_ext.insert(0, '*.ndat')
        self.form_input_file_button = tk.Button(left_frame, text='Form Input File', command=self.form_input_file)
        self.run_code_button = tk.Button(left_frame, text='Run Code', command=self.run_code)
        self.run_jackknife_button = tk.Button(left_frame, text='Run Jackknife', command=self.run_jackknife)

        self.text_area = tk.Text(mid_frame)
        self.save_button = tk.Button(mid_frame, text='Save Changes', command=self.save_changes)

        self.figure = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_frame)
        self.axes = self.figure.add_subplot(111)
        
        self.clear_plot_button = tk.Button(left_frame, text='Clear Plot', command=self.clear_plot)
        self.plot_obs_button = tk.Button(left_frame, text='Plot Obs', command=self.plot_obs)
        self.plot_result_button = tk.Button(left_frame, text='Plot Result', command=self.plot_result)
        self.plot_residuals_button = tk.Button(left_frame, text='Plot Residuals', command=self.plot_residuals)
        self.format_output_button = tk.Button(left_frame, text='Format Output', command=self.format_output)
        #self.normalise_button = tk.Button(left_frame, text='Normalise', command=self.normalise)


        self.change_path_button.pack(fill='x')
        self.n_spec_label.pack(fill='x')
        self.text_ext.pack(fill='x')
        self.form_input_file_button.pack(fill='x')
        self.run_code_button.pack(fill='x')
        self.run_jackknife_button.pack(fill='x')


        self.text_area.pack(fill='both', expand=True)
        self.save_button.pack(fill='x')


        #self.normalise_button.pack(side='bottom', fill='x')
        self.format_output_button.pack(side='bottom', fill='x')
        self.plot_residuals_button.pack(side='bottom', fill='x')
        self.plot_result_button.pack(side='bottom', fill='x')
        self.plot_obs_button.pack(side='bottom', fill='x')
        self.clear_plot_button.pack(side='bottom', fill='x')

        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, right_frame)
        toolbar.update()
        
        self.path =  './tests'
        self.extension = "*.ndat"
        
        self.path_fd3 = './execfd3/fd3'
        

    def change_path(self):
        directory_path = filedialog.askdirectory(initialdir='./tests')
        self.path = directory_path + "/"
        shutil.copy(self.path_fd3, self.path + "fd3")
        print(self.path + "fd3")

    def form_input_file(self):
        
        self.text_area.delete('1.0', tk.END)
        
        self.extension = self.text_ext.get()
        
        print(self.path)
        print(self.extension)
        
        files = fnmatch.filter(os.listdir(self.path), self.extension)

        self.n_spec_label.config(text=str(len(files)))

        bjd_df = pd.DataFrame(columns=['filename', 'BJD'])
        
        for nrm_file in files:
            fits_file = nrm_file.replace('.ndat', '.fits')
            with fits.open(self.path+fits_file) as hdul:
                bjd_df = bjd_df.append({'filename': nrm_file, 'BJD': hdul[0].header['BJD']}, ignore_index=True)
        
        
        bjd_df = bjd_df.sort_values('BJD')
        
        files = bjd_df['filename'].str.replace('.fits', '.ndat').tolist()
        
        old_text = self.n_spec_label.cget("text")
        self.n_spec_label.config(text=old_text + " " + str(len(files)))
        
        spectra = [pd.read_csv(self.path+f, names=['wavelength', 'flux'], delim_whitespace=True) for f in files]
        
        
        
        log_wavelength_min = np.max([np.log(df['wavelength'].min()) for df in spectra])
        log_wavelength_max = np.min([np.log(df['wavelength'].max()) for df in spectra])
        
        step = 0.020 # in A
        NN = int((np.e**log_wavelength_max-np.e**(log_wavelength_min))/step) # Number of pixels for resampling
        
        log_wavelength_step = (log_wavelength_max - log_wavelength_min) / NN
        
        common_log_wavelength = np.arange(log_wavelength_min, log_wavelength_max, log_wavelength_step)
        
        resampled_spectra = pd.DataFrame()
        resampled_spectra['log_wavelength'] = common_log_wavelength
        
        for i, df in enumerate(spectra):
            df['log_wavelength'] = np.log(df['wavelength'])
            
            f = interpolate.interp1d(df['log_wavelength'], df['flux'], fill_value="extrapolate")
            
            resampled_spectra[f'flux_{i}'] = f(common_log_wavelength)
            
            
        resampled_spectra = resampled_spectra.applymap(lambda x: f"{x:1.10f}")
        
        #np.savetxt(path+'resampled_spectra.dat', resampled_spectra.values, header=f'{resampled_spectra.shape[1]} X {resampled_spectra.shape[0]}', comments='# ')
        
        with open(self.path+'resampled_spectra.dat', 'w') as f:
            f.write(f'# {resampled_spectra.shape[1]} X {resampled_spectra.shape[0]}\n')
        resampled_spectra.to_csv(self.path+'resampled_spectra.dat', sep=' ', index=False, header=False, mode='a')
        
        
        bjd_df.to_csv(self.path+'BJD_values.csv', index=False, sep = "\t")
        
        
        
        formatted_bjd_df = bjd_df.copy()
        formatted_bjd_df['col2'] = 0.0
        formatted_bjd_df['col3'] = 0.5
        formatted_bjd_df['col4'] = 1.0
        formatted_bjd_df['col5'] = 1.0
        formatted_bjd_df = formatted_bjd_df.drop(columns=['filename'])
        
        formatted_bjd_df['BJD'] = formatted_bjd_df['BJD'].apply(lambda x: format(x - 2400000, '.5f'))
        formatted_bjd_df['col2'] = formatted_bjd_df['col2'].apply(lambda x: format(x, '.1f'))
        formatted_bjd_df['col3'] = formatted_bjd_df['col3'].apply(lambda x: format(x, '.1f'))
        formatted_bjd_df['col4'] = formatted_bjd_df['col4'].apply(lambda x: format(x, '.1f'))
        formatted_bjd_df['col5'] = formatted_bjd_df['col5'].apply(lambda x: format(x, '.1f'))
        
        with open(self.path+'formatted_BJD_values.dat', 'w') as f:
            f.write('\n'.join(formatted_bjd_df.apply(' '.join, axis=1)))
        
        
        with open(self.path+'input.in', 'w') as f:
            f.write("resampled_spectra.dat     temp.obs  1 1 0" + "\n")
            f.write("\n")
            
            f.write('\n'.join(formatted_bjd_df.apply(' '.join, axis=1)), )
            
            f.write("\n")
            f.write("\n")
            f.write("1 0 0 0 0 0  0 0  0 0  0 0" + "\n")
            f.write("\n")
            f.write("2.74 0.0 43563.5 0.1  0.02  0.003 217. 5.  78. 3.  155.  5.  0. 0." + "\n")
            f.write("\n")
            f.write("10 200 0.00001  temp.obs.mod  temp.obs.res  temp.obs.rvs" + "\n")
            f.write("\n")
            
            
        with open(self.path+'input.in', 'r') as file:
            file_content = file.read()

        self.text_area.insert('1.0', file_content)

    def run_code(self):
        
        executable_path = os.path.join(self.path, 'fd3')
        #command = './fd3 < input.in > output.l'
        command = f'{executable_path} < input.in > output.l'
        #print(command)
        #self.result = subprocess.run(command, shell=True, cwd=self.path)
        self.result = subprocess.Popen(command,shell=True,cwd=self.path, stdin=None,stdout=None,stderr=None,close_fds=True)

    def run_jackknife(self):
        aa = np.loadtxt(self.path+'resampled_spectra.dat', skiprows=1)
        num_rows, num_cols = aa.shape
        nc = num_cols-1
        
        header, data, footer = self.parse_file(self.path+'input.in')
        
        wl1 = float(header['value1'])
        wl2 = float(header['value2'])
        
        aa = np.loadtxt(self.path+'resampled_spectra.dat', skiprows = 1)
        df = pd.DataFrame(aa)
        
        df = df[df[0] <= wl2+0.1]
        df = df[df[0] >= wl1-0.1]
        
        with open(self.path+'resampled_spectra.buff', 'w') as f:
            f.write(f'# {df.shape[1]} X {df.shape[0]}\n')
        df.to_csv(self.path+'resampled_spectra.buff', sep=' ', index=False, header=False, mode='a')
        
        
        
        for ic in range(1,nc+1,1):
            self.form_jack_input(ic,nc)
        for ic in range(1,nc+1,1):
            self.run_jack_node(ic, nc)
        time.sleep(60)
        all_pars = []
        df1 = self.parse_output(self.path+'output.l')
        for ic in range(1,nc+1,1):
            df = self.parse_output(self.path+'output_'+str(ic)+'.l')
            print("???")
            print(ic)
            print(df)
            pp = np.array(df['value'].astype(float))
            all_pars.append([])
            for jp in pp:
                all_pars[ic-1].append(jp)
            
        all_pars = np.array(all_pars)
        print(all_pars)
        dfpp = pd.DataFrame(all_pars)
        dfpp.to_csv(self.path+'jackknife.out', sep=' ', index=False, mode='w')
        
        for i in range(len(all_pars[:,0])):
            if all_pars[i][-1] < all_pars[i][-2]:
                bb = all_pars[i][-1]
                all_pars[i][-1] = all_pars[i][-2]
                all_pars[i][-2] = bb
                if len(all_pars[0]) == 5:
                    all_pars[i][-3] = (all_pars[i][-3] - 180) % 360
        
            
        print(all_pars)
        dfpp = pd.DataFrame(all_pars)
        dfpp.to_csv(self.path+'jackknife_swap.out', sep=' ', index=False, mode='w')
        
        errs = []
        for i in range(len(all_pars[0])):
            er = self.jackerr(all_pars[:,i])
            if len(all_pars[0]) == 5:
                if i == 2:
                    er = self.jackerr_angle(all_pars[:,i])
            errs.append(er)
        print(errs)
        df1['errors'] = np.array(errs)
        print(df1)
        df1.to_csv(self.path+'errors.dat', sep=' ', index=False, mode='w')
            
        
    def save_changes(self):
        now = datetime.now()
        date_string = now.strftime("%Y%m%d_%H%M")
        shutil.copy(self.path+'input.in', self.path+'input_'+date_string+'.in')
        file_content = self.text_area.get('1.0', 'end')
        with open(self.path+'input.in', 'w') as file:
            file.write(file_content)

    def clear_plot(self):
        self.axes.clear()
        self.canvas.draw()

    def plot_obs(self):
        aa = np.loadtxt(self.path+'resampled_spectra.dat', skiprows=1)
        num_rows, num_cols = aa.shape
        nc = num_cols-1
        if nc <= 10:
            for i in range(1,num_cols,1):
                self.axes.plot(aa[:,0],aa[:,i]-(i-1)*0.1,lw=0.5)
        elif nc <= 30:
            for i in range(1,num_cols,3):
                self.axes.plot(aa[:,0],aa[:,i]-(i-1)*0.03,lw=0.5)
        elif nc <= 100:
            for i in range(1,num_cols,10):
                self.axes.plot(aa[:,0],aa[:,i]-(i-1)*0.01,lw=0.5)
        else:
            for i in range(1,num_cols,100):
                self.axes.plot(aa[:,0],aa[:,i]-(i-1)*0.001,lw=0.5)
        self.canvas.draw()
        
    def parse_file(self,file_name):
        with open(file_name, 'r') as file:
            lines = file.readlines()
            
            header = lines[0].strip()  
            footer = []
            data = []
            section = 'header'
        
            for line in lines[1:]:
                line = line.strip()  
                if line == '':
                    if section == 'header':
                        section = 'data'
                    elif section == 'data':
                        section = 'footer'
                    continue  
        
                if section == 'header':
                    header += '\n' + line
                elif section == 'data':
                    data.append(line.split())
                else:  
                    footer.append(line.split())
        
            header_parts = header.split()
            header_dict = {
                'name': header_parts[0],
                'value1': header_parts[1],
                'value2': header_parts[2],
                'type': header_parts[3],
                'flag1': header_parts[4],
                'flag2': header_parts[5],
                'flag3': header_parts[6]
            }
        
            footer_dict = {
                'flags': footer[0],
                'values': footer[1],
                'final_values': footer[2]
            }
                
        return header_dict, data, footer_dict

    def plot_result(self):
        
        filein = self.path+'input.in'
        header,data,footer = self.parse_file(filein)
        fname = footer['final_values'][-3]
        
        aa = np.loadtxt(self.path+fname, skiprows=1)
        self.axes.plot(aa[:,0],aa[:,1],"-k",lw=0.5)
        self.axes.plot(aa[:,0],aa[:,2],"-r",lw=0.5)
        self.canvas.draw()

    def plot_residuals(self):
        
        filein = self.path+'input.in'
        header,data,footer = self.parse_file(filein)
        fname = footer['final_values'][-2]
        
        aa = np.loadtxt(self.path+fname, skiprows=1)
        num_rows, num_cols = aa.shape
        nc = num_cols-1
        if nc <= 10:
            for i in range(1,num_cols,1):
                self.axes.plot(aa[:,0],aa[:,i]-(i-1)*0.1,lw=0.5)
        elif nc <= 30:
            for i in range(1,num_cols,3):
                self.axes.plot(aa[:,0],aa[:,i]-(i-1)*0.03,lw=0.5)
        elif nc <= 100:
            for i in range(1,num_cols,10):
                self.axes.plot(aa[:,0],aa[:,i]-(i-1)*0.01,lw=0.5)
        else:
            for i in range(1,num_cols,100):
                self.axes.plot(aa[:,0],aa[:,i]-(i-1)*0.001,lw=0.5)
        self.canvas.draw()
        
    def format_output(self):
        header, data, footer = self.parse_file(self.path+'input.in')
        fname = self.path + footer['final_values'][-3]
        aa = np.loadtxt(fname, skiprows = 1)
        aa[:,0] = np.e**aa[:,0]
        aa[:,1] += 0.5
        aa[:,2] += 0.5
        
        with open(self.path + footer['final_values'][-3][:-4] + ".formatted", "w") as out:
            for i in range(len(aa[:,0])):
                out.write(str(aa[i][0]) + "\t" + str(aa[i][1]) + "\t" + str(aa[i][2]) + "\n")
        for j in range(1,len(aa[0])):
            with open(self.path + footer['final_values'][-3][:-4] + ".comp" + str(j), "w") as out:
                for i in range(len(aa[:,0])):
                    out.write(str(aa[i][0]) + "\t" + str(aa[i][j]) + "\n")
        

    def normalise(self):
        print("Normalise")
        
    def form_jack_input(self, ic, nc):
    
        def save_file(file_name, header, data, footer):
            with open(file_name, 'w') as file:
                file.write(f"{header['name']} {header['value1']} {header['value2']} {header['type']} {header['flag1']} {header['flag2']} {header['flag3']}\n\n")
        
                for row in data:
                    file.write(' '.join(row) + '\n')
                
                file.write('\n')  
        
                file.write(' '.join(footer['flags']) + '\n\n')
                file.write(' '.join(footer['values']) + '\n\n')
                file.write(' '.join(footer['final_values']) + '\n')
    
            
        
        header, data, footer = self.parse_file(self.path+'input.in')

        aa = np.loadtxt(self.path+'resampled_spectra.buff', skiprows = 1, usecols = [c for c in range(nc+1) if c != ic])
        df = pd.DataFrame(aa)
        
        with open(self.path+'resampled_spectra_'+str(ic)+'.dat', 'w') as f:
            f.write(f'# {df.shape[1]} X {df.shape[0]}\n')
        df.to_csv(self.path+'resampled_spectra_'+str(ic)+'.dat', sep=' ', index=False, header=False, mode='a')
        
        
        del data[ic-1]
        
        header['name'] = 'resampled_spectra_'+str(ic)+'.dat'
        header['type'] = 'temp_'+str(ic)+'.obs'
        
        
        footer['final_values'][-3] = footer['final_values'][-3][:-4] + "_" + str(ic) + footer['final_values'][-3][-4:]
        footer['final_values'][-2] = footer['final_values'][-2][:-4] + "_" + str(ic) + footer['final_values'][-2][-4:]
        footer['final_values'][-1] = footer['final_values'][-1][:-4] + "_" + str(ic) + footer['final_values'][-1][-4:]
        
        save_file(self.path+'input_'+str(ic)+'.in', header, data, footer)    
            
        

    def run_jack_node(self, ic, nc):
        shutil.copy(self.path_fd3, self.path + "fd3_"+str(ic))
        command = './fd3_' +str(ic)+ ' < input_' +str(ic)+ '.in' + ' > output_' +str(ic)+ '.l'
        if ic%9 == 0 or ic == nc:
            self.result = subprocess.run(command, shell=True, cwd=self.path)
        else:    
            self.result = subprocess.Popen(command,shell=True,cwd=self.path, stdin=None,stdout=None,stderr=None,close_fds=True)
            
            
    def jackerr(self,array_par):
        print(array_par)
        nc = len(array_par)
        ss = sum(array_par)/float(nc)
        se = 0.0
        for i in range(nc):
            se += (array_par[i]-ss)**2
        se *= float(nc-1)/float(nc)
        se = np.sqrt(se)
        print(se)
        return se
    
    def normalize_angle(self,angle):
        return angle % 360
    
    def circular_mean(self,angles):
        angles_rad = np.deg2rad(angles)  # Convert to radians
        return np.rad2deg(np.arctan2(np.mean(np.sin(angles_rad)), np.mean(np.cos(angles_rad)))) % 360
    
    def angle_difference(self,angle1, angle2):
        return 180 - abs(abs(angle1 - angle2) - 180)
    
    def jackerr_angle(self, array_par):
        array_par = [self.normalize_angle(a) for a in array_par]
        nc = len(array_par)
        mean_angle = self.circular_mean(array_par)
        se = 0.0
        for i in range(nc):
            diff = self.angle_difference(array_par[i], mean_angle)
            se += diff ** 2
        se *= float(nc - 1) / float(nc)
        se = np.sqrt(se)
        print(se)
        return se
    
    def parse_output(self, fpath):
        start_reading = False
        data = []
        flag = False
        with open(fpath) as file:  
            for line in file:
                if not flag:
                    if "converged disentangling runs" in line:
                        flag = True
                    continue
                if "gof=" in line:
                    if "starting point" in line:
                        continue
                    start_reading = True
                    data = []  
                    continue
                if start_reading and line.strip() == "":
                    continue
                if "completed" in line:
                    break
                if start_reading:
                    print("!!!", line)
                    parameter, value = line.strip().split("=")
                    print("!!!", parameter, value)
                    try:
                        value = float(re.findall("[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", value)[0])
                        print(value)
                        if "periast long" in line:
                            value = value % 360
                        data.append({"parameter name": parameter.strip(), "value": value})
                        print(value)
                    except:
                        print(line)
    
        df = pd.DataFrame(data)
        print(df)
        return df
        
    
    
def main():
    app = Application()
    app.mainloop()

if __name__ == "__main__":
    main()