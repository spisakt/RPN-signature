library(oro.nifti)
source('~/src/R.lib/custom.ggplot.r')

recalculate=T
RSN=T
if (recalculate)
{
  if (RSN==T)
    atlas=readNIfTI("data/atlas_modules.nii.gz")
  else
    atlas=readNIfTI("data/atlas_relabeled.nii.gz")
  
  atlas_labels=read.table("data/atlas_relabeled.tsv")
  
  #modules=atlas
  
  map=readNIfTI("~/projects/PAINTeR/comp/bochum_trace_back_regs/SIGNED_REGIONAL_PREDICTIVE_WEIGHT_THR.nii.gz")
  map2=readNIfTI("~/projects/PAINTeR/src/res/bingel_2011_pain_matrix_ptfce.nii.gz")
  
  labels=sort(unique(as.numeric(atlas)))
  if (RSN==T)
  {
    regs=unique(atlas_labels$modules)[labels]
    group=regs
  }
  else
  {
    regs=atlas_labels$labels[labels]
    group=atlas_labels$modules[labels]
  }
    
  modules=atlas
  
  value1=c()
  value2=c()
  
  for (l in labels)
  {
    if (l==0)
      next;
    #modules[modules==l]=as.numeric(as.factor(atlas_labels$modules)[l])
    print(l)
    value1[l]=(mean(map[atlas==l]))
    value2[l]=mean(map2[atlas==l])
  }
  
  data.orig=data.frame(group=group, value1=value1, value2=value2, region=regs)
  #writeNIfTI(modules, "data/atlas_modules")
}

d=data.frame(RSN=factor(c(as.character(data.orig$group),
                             as.character(data.orig$group)),
                           levels=c("DMnet", "VISnet", "CER", "VATTnet_SALnet_BG_THAL",
                                                                  "LIMnet", "FPnet_VISDN",  "MOTnet")
                           ), VAL=c(data.orig$value1/max(data.orig$value1)*max(data.orig$value2), data.orig$value2), MAP=as.factor(c(rep("myRPN", 7), rep("PainMatrix", 7))))

#levels = list( "DMnet", "VISnet", "CER", "VATTnet_SALnet_BG_THAL",
#                              "LIMnet", "FPnet_VISDN",  "MOTnet"  ))

print(cor(abs(data.orig$value1), data.orig$value2))

p=ggplot(data=d, aes(x=RSN, group=MAP, fill=MAP, color=MAP))+
  geom_line( aes(y=VAL))+
  #coord_polar()+
  theme(legend.position = "none",
        #axis.text = element_blank(),
        axis.title = element_blank(),
        #panel.grid = element_blank(),
        #plot.margin = unit(rep(-1,4), "cm")
        )+
  ylim(min(d$VAL)-1,max(d$VAL))


# # Make the plot
# p = ggplot(data, aes(x=id, fill=group)) +       # Note that id is a factor. If x is numeric, there is some space between the first bar
#   geom_bar(aes(y=abs(value1)), stat="identity", alpha=0.5, size=2, fill="blue") +
#   geom_bar(aes(y=abs(value2)), stat="identity", alpha=0.5, size=2, fill="red") +
#   ylim(-max(value1)*10,max(value1)*2) +
#   theme_minimal() +
#   theme(
#     legend.position = "none",
#     axis.text = element_blank(),
#     axis.title = element_blank(),
#     #panel.grid = element_blank(),
#     plot.margin = unit(rep(-1,4), "cm")
#   ) +
#   coord_polar() +
#   geom_text(data=label_data, aes(x=id, y=1, label=region, hjust=hjust), color="black", fontface="bold",alpha=0.6, size=2.5, angle= label_data$angle, inherit.aes = FALSE )

print(p)