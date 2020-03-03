import { Component, OnInit,Injectable, Input } from '@angular/core';
import { ApiService } from '../../api-service/api.service';
import { ToastrManager } from 'ng6-toastr-notifications';
import { Router } from '@angular/router';
import { Ng4LoadingSpinnerService } from 'ng4-loading-spinner';
@Component({
  selector: 'app-import-model',
  templateUrl: './import-model.component.html',
  styleUrls: ['./import-model.component.css']
}) 
@Injectable({ providedIn: 'root' })
export class ImportModelComponent implements OnInit {
  @Input() header;
  file:any;
  files;
  classCount;
  fileName:string;
  counter = Array;
  modelSelect:boolean=false;
  class={};
  changeFile:boolean=false;
  objectKeys=Object.keys
  importClasses;
  errorFileUpload:boolean=false;
  classCountCheck;
  importClassesCheck={};
  fileNameCheck;
  extension;
  constructor(private apiservice: ApiService,private toastr:ToastrManager,private router: Router,private spinnerService: Ng4LoadingSpinnerService) { 
    // this.router.routeReuseStrategy.shouldReuseRoute = function(){
    //   return false;
    // }
  }

  ngOnInit() {
    if(this.header){
      this.header=JSON.parse(this.header);
      this.classCount=this.header["importClassCount"]
      this.classCountCheck=this.classCount
      this.importClasses = this.header["importClasses"];
      this.fileName = this.header["importFolder"];
      this.fileNameCheck = this.header["importFolder"];
      this.class={}
      for(let i=0;i<Object.keys(this.importClasses).length;i++){
        // this.class={}
        this.class[i]=this.importClasses[i]
        this.importClassesCheck[i] = this.importClasses[i]
            }
    }

    
  }
  upload(files:File){
    // var e= files;
    //pick from one of the 4 styles of file uploads below
    // this.file = e.target.files[0];
    this.spinnerService.show();
    let timeNow = new Date()
    console.log("--==--==--==--==--==--==--==--=="+timeNow.getHours()+ ":" + timeNow.getMinutes() + ":" + timeNow.getSeconds())
    const formData: FormData = new FormData();
    formData.append('file', files[0], files[0].name);
    this.files = files;
    this.fileName = files[0].name
    this.extension = this.fileName.split(".")[1]
    if(this.extension!="zip"){
      this.toastr.errorToastr("Please Upload Zip File Only!", "Error")
      this.errorFileUpload=true;
    }
    else{
    this.changeFile=false
    this.apiservice.getUnzippedFiles(formData).subscribe(
      success => {
        this.spinnerService.hide();
        let timeNow2 = new Date()
        console.log("--==--==--==--==--==--==--==--=="+timeNow2.getHours()+ ":" + timeNow2.getMinutes() + ":" + timeNow2.getSeconds())
        console.log("result from backend :",success);
      },
      err =>{
        this.spinnerService.hide();
        console.log(err);
      }
      
    );
    }
    // console.log(typeof files)
    // console.log(files.length)
    // for(let each_file=0;each_file<files.length;each_file++){
    //   console.log(files[0][0])
    //   console.log(typeof files)
    // }
    
  }
  fileChanged(e) {
    this.files = e.target.files[0];
    console.log(this.files)
    var filesArray=[];
    for (var i = 0; i < this.files.length; i++) {
      filesArray.push(this.files[i]);
    }
    console.log(filesArray)
}

uploadDocument() {
  console.log(this.classCount);
}
changeDataset(){
  this.changeFile=true;
  this.fileName=null;
}
onClickContinue(){
  let count=0;let check=0;
  let tempClass = this.class;
  this.class={}
  for(let i=0;i<this.classCount;i++){
    this.class[i]=tempClass[i]
  }
  for(let classId=0;classId<this.classCount;classId++){
    if(this.class[classId]){
      count = count+1;
    }
  }
  console.log(this.class,this.classCount);
  if((count!=this.classCount)){
  this.toastr.errorToastr('Enter All The Classes', );
  }
  if(!(this.fileName)){
    this.toastr.errorToastr('Upload Dataset', );
    // this.toastr.warningToastr('Upload Dataset', );
  }
  if(this.extension != "zip"){
    this.toastr.errorToastr("Selected File is Not a Zip File", "Error!")
  }
  else if((count==this.classCount) && (this.fileName) && (this.extension == 'zip')){
    this.modelSelect=true;
    console.log(this.class)
    
    if(window.location.href.includes("modelSelect")){
      console.log(this.classCount,this.classCountCheck,this.class)
        if(this.classCountCheck==this.classCount && this.fileName==this.fileNameCheck){
          for(let classId=0;classId<this.classCount;classId++){
            if(this.class[classId]==this.importClassesCheck[classId]){
              check = check+1
            }
          }
          
        }      
    }
  
    if(check==this.classCount){
      window.location.reload();
    }
    else{
    this.router.navigate(['/show'],{ queryParams: {'modelSelect': JSON.stringify(this.modelSelect),
                                                    'importFolder':JSON.stringify(this.fileName),
                                                   'importclassCount':JSON.stringify(this.classCount),
                                                   'importClasses':JSON.stringify(this.class)} });
    }// this.router.navigated=false;
    console.log("next")
    // window.location.reload();
  }
  
}
}

