df = read.csv("res/predictive_connections.csv")

modul = c(structure(as.character(df$mod_A), names=as.character(df$reg_A)), structure(as.character(df$mod_B), names=as.character(df$reg_B)))
modul = modul[!duplicated(names(modul))]
modul = modul[order(modul, names(modul))]

mod_colors=c(
  rgb(171/255, 112/255, 83/255), # CER
  rgb(0, 0, 0), # GS
  rgb(244/255, 160/255,85/255), # VA_S_B_T
  rgb(119/255,54/255,159/255), # LIM
  rgb(204/255, 65/255, 96/255), # FP
  rgb(153/255,244/255,121/255), # MOT
  rgb(87/255,102/255, 237/255), #DMN
  rgb(255/255,250/255,115/255) #VIS
)

modul_color = structure(mod_colors[1:8], names = unique(modul))
reg_color = structure(mod_colors[as.factor(modul)], names = names(modul))

library(circlize)
gap.after = do.call("c", lapply(table(modul), function(i) c(rep(2, i-1), 15)))

circos.clear()
circos.par(gap.after = gap.after, cell.padding = c(0, 0, 0, 0))

cols=c(rgb(63/255, 127/255, 147/255), rgb(209/255, 72/255, 58/255))

chordDiagram(df[, c("reg_A", "reg_B", "weight")], order = names(modul), grid.col = reg_color,
              directional = 0, annotationTrack = "grid", annotationTrackHeight=0.1, preAllocateTracks = list(
                list(track.height = 0.1)), col=cols[(sign(df$weight)+3)/2]#, link.border = cols[(sign(df$weight)+3)/2]
             )

 circos.track(track.index = 2, panel.fun = function(x, y) {
      xlim = get.cell.meta.data("xlim")
      ylim = get.cell.meta.data("ylim")
      sector.index = get.cell.meta.data("sector.index")
      circos.text(mean(xlim), mean(ylim), sector.index, col = "black", cex = 0.6, facing = "inside", niceFacing = TRUE)
  }, bg.border = NA)
 
 #modul=reg
  # tmp=modul
  # modul=names(modul)
  # names(modul)=modul
  # for(m in modul) {
  #      reg = names(modul[modul == m])
  #      highlight.sector(sector.index = reg, track.index = 1, col = modul_color[m], 
  #                       text = m, text.vjust = -1, niceFacing = TRUE)
  # }
circos.clear()