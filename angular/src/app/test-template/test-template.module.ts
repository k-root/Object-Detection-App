import { NgModule } from '@angular/core';
import { RouterModule } from '@angular/router';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';


import { TestTemplateRoutes } from './test-template.routing';
import { TestTemplateComponent } from './test-template.component';
import { PdfViewerModule } from 'ng2-pdf-viewer';

import { LearnModeModule } from '../learn-mode/learn-mode.module'


@NgModule({
  declarations: [TestTemplateComponent],
  imports: [
    CommonModule,
    RouterModule.forChild(TestTemplateRoutes),
    FormsModule,
    PdfViewerModule,
    LearnModeModule
  ]
})
export class TestTemplateModule { }
