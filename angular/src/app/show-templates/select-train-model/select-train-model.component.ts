import { Component, OnInit,Input } from '@angular/core';
import { Router } from '@angular/router';
import { ToastrManager } from 'ng6-toastr-notifications';
import { ApiService } from '../../api-service/api.service';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
@Component({
  selector: 'app-select-train-model',
  templateUrl: './select-train-model.component.html',
  styleUrls: ['./select-train-model.component.css']
})
export class SelectTrainModelComponent implements OnInit {
  @Input() header;
  modelSelect:boolean=false;
  trainSelect:boolean=false;
  selectedModel;
  currentModel;
  epoch;
  learningRate;
  stepsPerEpoch:number = 128;
  objectKeys=Object.keys
  importClasses; importclassCount;
  selectModelName;
  importFolder;
  enableNext:boolean=false;
  globalStr = "from select train";
  modelTraining;
  models=["Model 1","Model 2","Model 3"]
  constructor(private router: Router , private toastr:ToastrManager , private apiservice :ApiService) { }

  ngOnInit() {
    console.log(this.header,"-----------------------------------------------------------");
    console.log(this.globalStr);
    this.header=JSON.parse(this.header)
    if(Object.keys(this.header).includes("modelTraining")){
      this.modelTraining = this.header["modelTraining"];
    }
    if(Object.keys(this.header).includes("importClasses")){
      this.importClasses = this.header["importClasses"];
    }
    if(Object.keys(this.header).includes("importClassCount")){
      this.importclassCount = this.header["importClassCount"];
    }
    if(Object.keys(this.header).includes("selectModelName")){
      this.selectModelName = this.header["selectModelName"];
    }
    if(Object.keys(this.header).includes("importFolder")){
      this.importFolder = this.header["importFolder"];
    }
    
    if(this.header["select"]=="chooseModel"){
      this.modelSelect=true;
    }
    else if(this.header["select"]=="chooseTrain"){
      this.trainSelect=true;     
  }
  this.apiservice.getModelNames().subscribe(
    resp => {
      // console.log(typeof resp , resp , typeof this.models)
      this.models = resp
      console.log("result from backend :",resp);
      
    }
    
    
  );
  
}
  onSelectModel(index){
    this.enableNext=true;
    if(this.currentModel+1){
      document.getElementById(this.currentModel).style.backgroundColor="";
      document.getElementById(this.currentModel).style.color="grey";
    }
    console.log(this.currentModel)
    this.currentModel=index;
    document.getElementById(index).style.backgroundColor="#e2e3ea";
    document.getElementById(index).style.color="black";
}
  onClickContinue(){
    if(this.trainSelect)
    {
    this.router.navigate(['/show'],{ queryParams: {'evaluateSelect': JSON.stringify(true),'epochs':JSON.stringify(this.epoch),'learningRate':JSON.stringify(this.learningRate), 'stepsPerEpoch':JSON.stringify(this.stepsPerEpoch)} });
    }
    else{
     this.router.navigate(['/show'],{ queryParams: {'trainSelect': JSON.stringify(true),'selectModelName':JSON.stringify(this.models[this.currentModel])} }); 
    }
  }
  onClickNext(){
    let trainview:boolean=true
    if(this.learningRate>1){
      this.toastr.warningToastr('Set Learning Rate to < 1', );
      trainview=false;
    }
    
    if(!(this.importFolder || this.importclassCount || this.importClasses)){
      this.toastr.warningToastr('Import Dataset', );
      trainview=false;
    }
    if(!this.selectModelName){
      this.toastr.warningToastr('Select Model', );
      trainview=false;
    }
    if(trainview){
      this.enableNext=true;
      this.modelTraining=true;

      
      if(this.trainSelect)
      {
         this.router.navigate(['/show'],{ queryParams: {'evaluateSelect': JSON.stringify(true),'epochs':JSON.stringify(this.epoch),'learningRate':JSON.stringify(this.learningRate),'modelTraining':JSON.stringify(this.modelTraining)} });
      }
    }
  }
}