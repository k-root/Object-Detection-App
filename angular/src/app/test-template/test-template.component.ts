import { Component, OnInit } from '@angular/core';
import { ApiService } from '../api-service/api.service';
import { Ng4LoadingSpinnerService } from 'ng4-loading-spinner';
import { DataShareService } from '../data-share/data-share.service';
import { ActivatedRoute } from '@angular/router';
import { DomSanitizer } from '@angular/platform-browser';
import Swal from 'sweetalert2';

declare var require: any;
declare var $: any;

@Component({
  selector: 'app-test-template',
  templateUrl: './test-template.component.html'
})
export class TestTemplateComponent implements OnInit {
  templates: String[];
  selectedTemplate: string;
  showResult = false;
  document: any;
  fileToUpload: File = null;
  resultData: any;
  fileName: string;
  objKeys = Object.keys;
  imageAlt = '';
  imageLoading = false;
  tables = {};
  showTable=false;
  column_length = 0;
  selectedImage:string;
  debugOutput: any;

  sampleImagesDir = { 'invoice':[], 'quote':[], 'license': [], 'passport':[], 'default':[] };

  templateType:string="";
  constructor(
    private apiService: ApiService,
    private spinnerService: Ng4LoadingSpinnerService,
    private _Activatedroute: ActivatedRoute,
    private dataService: DataShareService,
    private domSanitizer: DomSanitizer
  ) {}

  ngOnInit() {
    this.dataService.changeMessage("Test Model")
    
  }

  

}
