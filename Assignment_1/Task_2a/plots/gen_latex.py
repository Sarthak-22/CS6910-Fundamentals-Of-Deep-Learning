def getsubString(name):
    stri="\\newpage\n"
    stri+= f"\\subsection{{"+name +f"}}\n"
    return stri
def getImageString(filename,caption,scale):
    stri = f"\\begin{{subfigure}}{{.5\\textwidth}}\n"
    stri+= "\\centering\n"
    #stri+= "\\includegraphics[scale = "+str(scale)+f"]{{"+filename +f"}}\n"
    stri += "\includegraphics[width=1.2\\textwidth,height=1.2\\textheight,keepaspectratio]{" + filename+ "}"
    stri+= f"\\caption{{" + caption +f"}}\n"

    stri += f"\\end{{subfigure}}\n"

    
    return stri
def getImageString2(filename,caption,scale):
    
    stri = f"\\begin{{figure}}[H]\n"
    stri+= f"\\caption{{" + caption +f"}}\n"
    #stri+= "\\centering\n"
    stri+= "\\includegraphics[scale = "+str(scale)+f",left]{{"+filename +f"}}\n"
    stri += f"\\end{{figure}}\n"
    return stri

epochs = [1,2,10,50,360]
latex = ""

for epoch in epochs:
    latex += f"\\begin{{figure}}[H]\n"  

    #latex+=getsubString(f"Epoch {epoch}")
    for layer in [1,2]:
        nodes = []
        if (layer == 1):
            nodes = [1,2,3,4]
        else:
            nodes = [1,2]
        for node in nodes:
            fname = f"task2a/epoch{str(epoch)}_hl{str(layer)}{str(node)}.png"
            latex+=getImageString(fname,f"Plot for Hidden Layer {layer}, Node {node} (epoch: {epoch})",0.7)
    fname = f"task2a/epoch{str(epoch)}_output.png"
    latex+="\\pagebreak"
    latex += f"\\end{{figure}}\n"
    
    latex+=getImageString2(fname,f"Plot for Output Layer  (epoch: {epoch})",1.3)
    latex+="\\pagebreak"
print(latex)
    