import { Component, OnInit, Input, OnChanges } from '@angular/core';

import { ApiService } from '../api-service/api.service';
import { Ng4LoadingSpinnerService } from 'ng4-loading-spinner';


declare var $: any;

@Component({
  moduleId: module.id,
  selector: 'learn-mode',
  templateUrl: './learn-mode.component.html',
  styleUrls: ['./learn-mode.component.css']
})
export class LearnModeComponent implements OnInit {

  @Input() templateName;
  objKeys = Object.keys;
  learntTemplateExist = false;
  learntTemplateData;
  oldTemplateData;
  isData = false;
  output;
  constructor(private apiService: ApiService, private spinnerService: Ng4LoadingSpinnerService, ) { }

  ngOnInit() {
    console.log(this.templateName);
    this.spinnerService.show();
    this.apiService.getLearnTemplate(this.templateName).subscribe(
       results => {
        if (results) {
          this.learntTemplateData = results;
          console.log(this.learntTemplateData);
          this.learntTemplateExist = true;
          this.spinnerService.hide();
        }
      },
      error => {
        this.spinnerService.hide();
      }
    )

    console.log(this.learntTemplateExist)
  }


  updateOldValuesInLearnTemplate() {
    let pages = Object.keys(this.output);
    pages.forEach((page: string) => {
      let temp = this.learntTemplateData['document_template']['keyword_mapping'][page];
      let tempOutput = this.output[page];
      let length1 = tempOutput.length;
      let length2 = temp.length;

      console.log(length1,length2);

      if (length1 == length2) {
        for (let i = 0; i < length1; i++) {
          if (tempOutput[i].key == temp[i].keyword) {
            temp[i].old_value = tempOutput[i].value;
          }
          else {
            temp[i].old_value = '';
          }
        }
      }
      else if (length1 > length2) {
        let j=0;
        for (let i = 0; i < length1; i++) {
          for (j = 0; j < length2; j++) {
            if (tempOutput[i].key == temp[j].keyword) {
              temp[j].old_value = tempOutput[i].value;
            }
          }
        }
        while (j != length1) {
          temp[j] = {
            'keyword': tempOutput[j].key,
            'old_value': tempOutput[j].value,
            'learnt_value': '',
            'rule': ''
          }
          j++;
        }
      }
      else {
        let j=0;
        for (let i = 0; i < length2; i++) {
          let flag = false;
          for (j = 0; j < length1; j++) {
            if (tempOutput[j].key == temp[i].keyword) {
              temp[i].old_value = tempOutput[j].value;
              flag = true;
              break;
            }
          }
          if(!flag){
            temp[i].old_value = '';
          }
        }
      }
    });
    this.isData = true;
    console.log('updated', this.learntTemplateData);

  }

  createLearnTemplate() {
    let keywordMapping = this.learntTemplateData['document_template']['keyword_mapping'];
    let newKeywordMapping = {};
    let pages = Object.keys(this.output);
    pages.forEach((page: string) => {
      newKeywordMapping[page] = [];
      this.output[page].forEach(response => {
        newKeywordMapping[page].push({
          'keyword': response.key,
          'old_value': response.value,
          'learnt_value': '',
          'rule': ''
        });
      });
    });
    this.learntTemplateData['document_template']['keyword_mapping'] = newKeywordMapping;
    this.isData = true;
    console.log(this.learntTemplateData);
  }

  
  saveResults() {
    console.log(this.learntTemplateData, 'before saving');
    this.updateRuleInLearntTemplate();
    this.spinnerService.show();
    this.apiService.updateLearnTemplate(this.templateName, this.learntTemplateData).subscribe(
      result => {
        console.log('updated learn template', result);
        this.hideModel();
        this.showNotification('top', 'right', 'info', 'learn mode has been updated');
        this.spinnerService.hide();
      },
      error => {
        console.log('unable to save learn template');
        this.spinnerService.hide();
      }
    )
  }

  updateRuleInLearntTemplate(){
    let pages = Object.keys(this.learntTemplateData);
    pages.forEach((page:string) => {
      this.learntTemplateData[page] = this.learntTemplateData[page].map(
        element => {
          if(element.learnt_value.toString() && !element.rule){
            element.rule = 'replace';
          }
          return element;
        }
      );
    });
  }


hideModel(){
    $("#learnModal").modal("hide"); 
}

  showNotification(from, align, type, message) {
    // const type = ['', 'info', 'success', 'warning', 'danger'];

    $.notify(
      {
        icon: 'ti-gift',
        message: message
      },
      {
        type: type,
        timer: 4000,
        placement: {
          from: from,
          align: align
        },
        template:
          '<div data-notify="container" class="col-11 col-md-4 alert alert-{0} alert-with-icon" role="alert"><button type="button" aria-hidden="true" class="close" data-notify="dismiss"><i class="nc-icon nc-simple-remove"></i></button><span data-notify="icon" class="nc-icon nc-bell-55"></span> <span data-notify="title">{1}</span> <span data-notify="message">{2}</span><div class="progress" data-notify="progressbar"><div class="progress-bar progress-bar-{0}" role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100" style="width: 0%;"></div></div><a href="{3}" target="{4}" data-notify="url"></a></div>'
      }
    );
  }
}
