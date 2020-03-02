import { Routes } from '@angular/router';

import { AdminLayoutComponent } from './layouts/admin/admin-layout.component';
import { AuthLayoutComponent } from './layouts/auth/auth-layout.component';
import { AuthGuard } from "./gaurds/auth.guard";
import {SelectTrainModelComponent} from "./show-templates/select-train-model/select-train-model.component"
import { ShowTemplatesComponent } from './show-templates/show-templates.component';

export const AppRoutes: Routes = [
  
  {path:'selectModel',component:SelectTrainModelComponent},
  {
    path: '',
    redirectTo: 'login',
    pathMatch: 'full'
  },
  {
    path: '',
    component: AdminLayoutComponent,
    canActivate: [AuthGuard],
    children: [
      // {
      //   path: 'show',
      //   loadChildren:
      //     './show-templates/show-templates.module#ShowTemplatesModule'
      // },
      { path: 'show', component: ShowTemplatesComponent},
      {
        path: 'test',
        loadChildren: './test-template/test-template.module#TestTemplateModule'
      },
      {
        path: 'test/:name',
        loadChildren: './test-template/test-template.module#TestTemplateModule'
      },
      // {
      //   path:'',
      //   loadChildren:'./select-train-model/select-train-model.module#SelectTrainModelModule'
      // }
      // {
      //     path: 'forms',
      //     loadChildren: './forms/forms.module#Forms'
      // },{
      //     path: 'tables',
      //     loadChildren: './tables/tables.module#TablesModule'
      // },{
      //     path: 'maps',
      //     loadChildren: './maps/maps.module#MapsModule'
      // },{
      //     path: 'charts',
      //     loadChildren: './charts/charts.module#ChartsModule'
      // },{
      //     path: 'calendar',
      //     loadChildren: './calendar/calendar.module#CalendarModule'
      // },{
      //     path: '',
      //     loadChildren: './userpage/user.module#UserModule'
      // },{
      //     path: '',
      //     loadChildren: './timeline/timeline.module#TimelineModule'
      // },{
      //     path: '',
      //     loadChildren: './widgets/widgets.module#WidgetsModule'
      // }
    ]
  },
  {
    path: '',
    component: AuthLayoutComponent,
    children: [
      {
        path: 'login',
        loadChildren: './login/login.module#LoginModule'
      }
    ]
  }
];
