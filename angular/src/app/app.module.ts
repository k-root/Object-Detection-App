import { NgModule } from '@angular/core';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { RouterModule } from '@angular/router';
import { HttpModule } from '@angular/http';
import { APP_BASE_HREF } from '@angular/common';
import { FormsModule, NgModel } from '@angular/forms';
import { NgbModule } from '@ng-bootstrap/ng-bootstrap';
import {ToastrModule} from 'ng6-toastr-notifications'
import { AppComponent }   from './app.component';

import { SidebarModule } from './sidebar/sidebar.module';
import { FixedPluginModule } from './shared/fixedplugin/fixedplugin.module';
import { FooterModule } from './shared/footer/footer.module';
import { NavbarModule} from './shared/navbar/navbar.module';
import { AdminLayoutComponent } from './layouts/admin/admin-layout.component';
import { AuthLayoutComponent } from './layouts/auth/auth-layout.component';
import { AppRoutes } from './app.routing';
import { HttpClientModule } from '@angular/common/http';
import { Ng4LoadingSpinnerModule } from 'ng4-loading-spinner';
import { PdfViewerModule } from 'ng2-pdf-viewer';
import { AuthGuard } from './gaurds/auth.guard';
import {SelectTrainModelComponent} from "./show-templates/select-train-model/select-train-model.component";
import {ImportModelComponent} from './show-templates/import-model/import-model.component';
import { ShowTemplatesComponent } from './show-templates/show-templates.component';
import {TestEvaluateModelComponent} from './show-templates/test-evaluate-model/test-evaluate-model.component'
import { templateSourceUrl } from '@angular/compiler';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';

// import { SelectTrainModelModule } from './show-templates/select-train-model/select-train-model.module';




@NgModule({
    imports:      [
        BrowserAnimationsModule,
        FormsModule,
        RouterModule.forRoot(AppRoutes,{
          useHash: true
        }),
        NgbModule.forRoot(),
        HttpModule,
        SidebarModule,
        NavbarModule,
        FooterModule,
        FixedPluginModule,
        HttpClientModule,
        Ng4LoadingSpinnerModule.forRoot(),
        ToastrModule.forRoot(),
        MatProgressSpinnerModule,
        // SelectTrainModelModule
    ],
    providers: [AuthGuard],
    declarations: [
        AppComponent,
        AdminLayoutComponent,
        AuthLayoutComponent,
        SelectTrainModelComponent,
        ImportModelComponent,
        ShowTemplatesComponent,
        TestEvaluateModelComponent
    ],
    bootstrap:    [ AppComponent ]
})

export class AppModule { }
